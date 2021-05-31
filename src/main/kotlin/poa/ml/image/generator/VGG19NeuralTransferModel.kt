package poa.ml.image.generator

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.PoolingType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex
import org.deeplearning4j.nn.layers.AbstractLayer
import org.deeplearning4j.nn.layers.FrozenLayerWithBackprop
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.workspace.ArrayType
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.all
import org.nd4j.linalg.indexing.NDArrayIndex.interval
import org.nd4j.linalg.lossfunctions.impl.LossL2

class VGG19NeuralTransferModel(
    h5Path: String,
    width: Long,
    height: Long,
    nChannels: Long,
) {

    private var inputGradient: INDArray? = null

    private val vgg19: ComputationGraph
    private val contentLayer = LayerInfo("block5_conv4", 25, 25, 512)
    private val stylesLayers = listOf(
        LayerInfo("block1_conv1", 400, 400, 64),
        LayerInfo("block2_conv1", 200, 200, 128),
        LayerInfo("block3_conv1", 100, 100, 256),
        LayerInfo("block4_conv1", 50, 50, 512),
        LayerInfo("block5_conv1", 25, 25, 512)
    )
    private val loss2 = LossL2()

    init {

        val modelHdf5Filename = ClassPathResource(h5Path).file.path
        val pretrained = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename)

        val preBuild = TransferLearning.GraphBuilder(pretrained)
            .fineTuneConfiguration(FineTuneConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED).build()
            )
            .setInputTypes(InputType.convolutional(
                height,
                width,
                nChannels,
                CNN2DFormat.NCHW
            ))
            .removeVertexKeepConnections("block5_pool")

        for (l in stylesLayers) {
            preBuild.addVertex(
                l.flattenName(),
                PreprocessorVertex(CnnToFeedForwardPreProcessor(l.height, l.width, l.nChannels)),
                l.name)
        }
        preBuild.addVertex(
            contentLayer.flattenName(),
            PreprocessorVertex(CnnToFeedForwardPreProcessor(contentLayer.height,
                contentLayer.width,
                contentLayer.nChannels)),
            contentLayer.name)

        vgg19 = preBuild
            .addLayer(
                "outputs",
                CustomOutputLayerConf(::score, ::gradient),
                *stylesLayers.toMutableList().apply { add(contentLayer) }.map { it.flattenName() }.toTypedArray()
            )
            .setOutputs("outputs")
            .build()

        vgg19.addListeners(ScoreIterationListener(1))

        for (layer in vgg19.layers) {
            if (layer is org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer) {
                (layer.layerConf() as SubsamplingLayer).poolingType = PoolingType.AVG
            }
            if (layer is org.deeplearning4j.nn.layers.normalization.BatchNormalization) {
                layer.layerConf().cnn2DFormat = CNN2DFormat.NCHW
            }
            if (layer is AbstractLayer<*>) {
                layer.layerConf().iDropout = null
            }
        }


        for (vertex in vgg19.vertices) {
            if (vertex is LayerVertex) {
                val field = vertex::class.java.getDeclaredField("layer")
                field.isAccessible = true
                field.set(vertex, FrozenLayerWithBackprop(vertex.layer))
            }
        }

        vgg19.vertices[1] = object : BaseWrapperVertex(vgg19.vertices[1]) {
            override fun doBackward(
                tbptt: Boolean,
                workspaceMgr: LayerWorkspaceMgr?,
            ): Pair<Gradient, Array<INDArray>> {
                val res = super.doBackward(tbptt, workspaceMgr)
                inputGradient = res.second[0].detach()
                return res
            }
        }
        vgg19.init()
    }

    fun getInputGradient(img: INDArray, label: INDArray): INDArray {
        vgg19.fit(arrayOf(img), arrayOf(label))
        return inputGradient!!
    }

    fun feedForward(img: INDArray): Map<String, INDArray> {
        return vgg19.feedForward(img, false)
    }

    private fun score(input: INDArray, labels: INDArray, m: LayerWorkspaceMgr): Double {
        val (content, style) = contentAndStyles(input)
        return loss2.computeScore(content.array, labels, ActivationIdentity(), Nd4j.ones(*labels.shape()), true)
            .div(content.info.flattenSize() * 4)
    }

    private fun gradient(input: INDArray, labels: INDArray, m: LayerWorkspaceMgr): INDArray {
        val (content, style) = contentAndStyles(input)
        val epsilon = m.create(ArrayType.ACTIVATION_GRAD, input.dataType(), *input.shape())
        val contentGradient = loss2.computeGradient(content.array,
            labels,
            ActivationIdentity(),
            Nd4j.ones(*content.array.shape()))
            .div(content.info.flattenSize() * 4)
        epsilon[all(), interval(content.beginIdx, content.tillIdx)].assign(contentGradient)
        return epsilon

    }


    private fun contentAndStyles(input: INDArray): kotlin.Pair<InputSlice, List<InputSlice>> {
        val styles = mutableListOf<InputSlice>()
        var beginIdx = 0L
        for (styleLayer in stylesLayers) {
            val till = beginIdx + styleLayer.flattenSize()
            val slice = input[all(), interval(beginIdx, till)]
            styles.add(InputSlice(slice, beginIdx, till, styleLayer))
            beginIdx = till
        }
        val slice = input[all(), interval(beginIdx, input.size(1))]
        val content = InputSlice(slice, beginIdx, input.size(1), contentLayer)
        return content to styles
    }

}


private data class LayerInfo(
    val name: String,
    val height: Long,
    val width: Long,
    val nChannels: Long,
) {
    fun flattenName() = "${name}_flattened"
    fun flattenSize() = height * width * nChannels

}

private data class InputSlice(
    val array: INDArray,
    val beginIdx: Long,
    val tillIdx: Long,
    val info: LayerInfo,
)


