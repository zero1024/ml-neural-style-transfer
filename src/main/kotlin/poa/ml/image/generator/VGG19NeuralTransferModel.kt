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
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.autodiff.loss.LossReduce
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.all
import org.nd4j.linalg.indexing.NDArrayIndex.interval
import kotlin.math.pow

class VGG19NeuralTransferModel(
    h5Path: String,
    width: Long,
    height: Long,
    nChannels: Long,
) {

    val alpha = 10.0
    val betta = 40.0
    private var inputGradient: INDArray? = null

    private val vgg19: ComputationGraph
    val contentLayer = ContentLayerInfo("block5_conv4", 25, 25, 512)
    val stylesLayers = listOf(
        StyleLayerInfo("block1_conv1", 400, 400, 64),
        StyleLayerInfo("block2_conv1", 200, 200, 128),
        StyleLayerInfo("block3_conv1", 100, 100, 256),
        StyleLayerInfo("block4_conv1", 50, 50, 512),
        StyleLayerInfo("block5_conv1", 25, 25, 512)
    )

    fun getLabelSize() = stylesLayers.toMutableList<LayerInfo>().apply { add(contentLayer) }
        .map { it.flattenSize() }
        .reduce { i1, i2 -> i1 + i2 }

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
            preBuild.addLayer(
                l.flattenName(),
                CustomLayerConf(),
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
                CustomOutputLayerConf(::score),
                *stylesLayers.toMutableList<LayerInfo>().apply { add(contentLayer) }.map { it.flattenName() }
                    .toTypedArray()
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

    fun toLabel(contentImg: INDArray, styleImg: INDArray): INDArray {
        val label2D = Nd4j.create(1, getLabelSize())

        val (contentSlice, styleSlices) = contentAndStyles()

        val styleActivations = vgg19.feedForward(styleImg, false)
        for (styleSlice in styleSlices) {
            val styleLabel = styleActivations[styleSlice.info.flattenName()]
            label2D[all(), interval(styleSlice.beginIdx, styleSlice.tillIdx)]
                .assign(styleLabel)
        }

        val contentActivations = vgg19.feedForward(contentImg, false)
        val contentLabel = contentActivations[contentSlice.info.flattenName()]
        label2D[all(), interval(contentSlice.beginIdx, contentSlice.tillIdx)]
            .assign(contentLabel)

        return label2D

    }

    private fun score(sameDiff: SameDiff, input: SDVariable, labels: SDVariable): SDVariable {
        val (content, styles) = contentAndStyles()

        var res = sameDiff.loss.meanSquaredError(
            sameDiff.applyMask(labels, content.mask),
            sameDiff.applyMask(input, content.mask),
            null,
            LossReduce.SUM)
            .mul(alpha)
            .div(content.info.flattenSize() * 4.0)

        for (style in styles) {
            val styleRes = sameDiff.loss.meanSquaredError(
                sameDiff.applyMask(labels, style.mask),
                sameDiff.applyMask(input, style.mask),
                null,
                LossReduce.SUM)
                .mul((style.info as StyleLayerInfo).weight)
                .mul(betta)
                .div((0.0 + style.info.height * style.info.width * style.info.nChannels).pow(2.0) * 4.0)
            res = res.add(styleRes)
        }

        return res
    }

    fun contentAndStyles(): kotlin.Pair<InputSlice, List<InputSlice>> {
        val styles = mutableListOf<InputSlice>()
        var beginIdx = 0L
        for (styleLayer in stylesLayers) {
            val till = beginIdx + styleLayer.flattenSize()
            val mask = Nd4j.zeros(1, getLabelSize())
            mask[all(), interval(beginIdx, till)].assign(1)
            styles.add(InputSlice(mask, beginIdx, till, styleLayer))
            beginIdx = till
        }
        val mask = Nd4j.zeros(1, getLabelSize())
        mask[all(), interval(beginIdx, beginIdx + contentLayer.flattenSize())].assign(1)
        val content = InputSlice(mask, beginIdx, beginIdx + contentLayer.flattenSize(), contentLayer)
        return content to styles
    }

}


abstract class LayerInfo(
    val name: String,
    val height: Long,
    val width: Long,
    val nChannels: Long,
) {
    fun flattenName() = "${name}_flattened"
    abstract fun flattenSize(): Long

}

class ContentLayerInfo(
    name: String,
    height: Long,
    width: Long,
    nChannels: Long,
) : LayerInfo(name, height, width, nChannels
) {
    override fun flattenSize() = height * width * nChannels
}

class StyleLayerInfo(
    name: String,
    height: Long,
    width: Long,
    nChannels: Long,
    val weight: Double = 0.2,
) : LayerInfo(name, height, width, nChannels
) {
    override fun flattenSize() = nChannels * nChannels
}

data class InputSlice(
    val mask: INDArray,
    val beginIdx: Long,
    val tillIdx: Long,
    val info: LayerInfo,
)


