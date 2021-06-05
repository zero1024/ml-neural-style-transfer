package poa.ml.image.generator.model

import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.conf.CNN2DFormat
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
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex.*
import org.nd4j.linalg.indexing.conditions.GreaterThan
import org.nd4j.linalg.indexing.conditions.LessThan
import poa.ml.image.generator.applyMask
import poa.ml.image.generator.model.conf.GramMatrixLayerConf
import poa.ml.image.generator.model.conf.SameDiffLambdaOutputLayerConf
import poa.ml.image.generator.model.layer.ContentNeuralTransferLayerInfo
import poa.ml.image.generator.model.layer.NeuralTransferLayerInfo
import poa.ml.image.generator.model.layer.StyleNeuralTransferLayerInfo
import poa.ml.image.generator.resize
import kotlin.math.pow

class VGG19NeuralTransferModel(
    vgg19path: String,
    mb: Long = 1,
    private val alpha: Double = 10.0,
    private val betta: Double = 10.0,
) : NeuralTransferModel {


    private val imageLoader = NativeImageLoader()
    private var inputGradient: INDArray? = null
    private val model: ComputationGraph

    private val width = 400L
    private val height = 400L
    private val nChannels = 3L
    private val contentLayer = ContentNeuralTransferLayerInfo("block5_conv4", 25, 25, 512)
    private val stylesLayers = listOf(
        StyleNeuralTransferLayerInfo("block1_conv1", 400, 400, 64),
        StyleNeuralTransferLayerInfo("block2_conv1", 200, 200, 128),
        StyleNeuralTransferLayerInfo("block3_conv1", 100, 100, 256),
        StyleNeuralTransferLayerInfo("block4_conv1", 50, 50, 512),
        StyleNeuralTransferLayerInfo("block5_conv1", 25, 25, 512)
    )

    init {

        val modelHdf5Filename = ClassPathResource(vgg19path).file.path
        val pretrained = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename)

        val preBuild = TransferLearning.GraphBuilder(pretrained)
            .fineTuneConfiguration(FineTuneConfiguration.Builder().build())
            .setInputTypes(InputType.convolutional(
                height,
                width,
                nChannels,
                CNN2DFormat.NCHW
            ))
            .removeVertexKeepConnections("block5_pool")

        for (l in stylesLayers) {
            preBuild.addLayer(
                l.flattenLayerName(),
                GramMatrixLayerConf(mb),
                l.name)
        }

        preBuild.addVertex(
            contentLayer.flattenLayerName(),
            PreprocessorVertex(CnnToFeedForwardPreProcessor(contentLayer.height,
                contentLayer.width,
                contentLayer.nChannels)),
            contentLayer.name)

        model = preBuild
            .addLayer(
                "outputs",
                SameDiffLambdaOutputLayerConf(::score),
                *getNeuralTransferLayerInfoList()
                    .map { it.flattenLayerName() }
                    .toTypedArray()
            )
            .setOutputs("outputs")
            .build()

        for (layer in model.layers) {
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

        for (vertex in model.vertices) {
            if (vertex is LayerVertex) {
                val field = vertex::class.java.getDeclaredField("layer")
                field.isAccessible = true
                field.set(vertex, FrozenLayerWithBackprop(vertex.layer))
            }
        }

        model.vertices[1] = object : BaseWrapperVertex(model.vertices[1]) {
            override fun doBackward(
                tbptt: Boolean,
                workspaceMgr: LayerWorkspaceMgr?,
            ): Pair<Gradient, Array<INDArray>> {
                val res = super.doBackward(tbptt, workspaceMgr)
                inputGradient = res.second[0].detach()
                return res
            }
        }
    }

    override fun scaleForModel(img: INDArray): INDArray {
        val res = imageLoader.resize(img, width.toInt(), height.toInt())
        res[all(), point(0), all(), all()].divi(255)
        res[all(), point(1), all(), all()].divi(255)
        res[all(), point(2), all(), all()].divi(255)
        return res
    }

    override fun rescaleBack(img: INDArray, width: Int, height: Int): INDArray {
        val res = imageLoader.resize(img, width, height)
        res[all(), point(0), all(), all()].muli(255)
        res[all(), point(1), all(), all()].muli(255)
        res[all(), point(2), all(), all()].muli(255)
        res.replaceWhere(Nd4j.create(*res.shape()).assign(255), GreaterThan(255))
        res.replaceWhere(Nd4j.create(*res.shape()).assign(0), LessThan(0))
        return res
    }

    override fun score() = model.score()

    override fun inputGradient(img: INDArray, label: INDArray): INDArray {
        model.fit(arrayOf(img), arrayOf(label))
        return inputGradient!!
    }

    override fun toLabel(contentImg: INDArray, styleImg: INDArray): INDArray {
        val label2D = Nd4j.create(1, getLabelSize())

        val (contentSlice, styleSlices) = contentAndStyles()

        val styleActivations = model.feedForward(styleImg, false)
        for (styleSlice in styleSlices) {
            val styleLabel = styleActivations[styleSlice.info.flattenLayerName()]
            label2D[all(), interval(styleSlice.beginIdx, styleSlice.tillIdx)]
                .assign(styleLabel)
        }

        val contentActivations = model.feedForward(contentImg, false)
        val contentLabel = contentActivations[contentSlice.info.flattenLayerName()]
        label2D[all(), interval(contentSlice.beginIdx, contentSlice.tillIdx)]
            .assign(contentLabel)

        return label2D

    }

    //=====PRIVATE=====//

    private fun getLabelSize() = getNeuralTransferLayerInfoList()
        .map { it.flattenLayerSize() }
        .reduce { i1, i2 -> i1 + i2 }

    private fun getNeuralTransferLayerInfoList() =
        stylesLayers.toMutableList<NeuralTransferLayerInfo>().apply { add(contentLayer) }


    private fun score(sameDiff: SameDiff, input: SDVariable, labels: SDVariable): SDVariable {
        val (content, styles) = contentAndStyles()

        var res = sameDiff.loss.meanSquaredError(
            sameDiff.applyMask(labels, content.mask),
            sameDiff.applyMask(input, content.mask),
            null,
            LossReduce.SUM)
            .mul(alpha)
            .div(content.info.flattenLayerSize() * 4.0)

        for (style in styles) {
            val styleRes = sameDiff.loss.meanSquaredError(
                sameDiff.applyMask(labels, style.mask),
                sameDiff.applyMask(input, style.mask),
                null,
                LossReduce.SUM)
                .mul((style.info as StyleNeuralTransferLayerInfo).weight)
                .mul(betta)
                .div((0.0 + style.info.height * style.info.width * style.info.nChannels).pow(2.0) * 4.0)
            res = res.add(styleRes)
        }

        return res
    }

    private fun contentAndStyles(): kotlin.Pair<InputSlice, List<InputSlice>> {
        val styles = mutableListOf<InputSlice>()
        var beginIdx = 0L
        for (styleLayer in stylesLayers) {
            val till = beginIdx + styleLayer.flattenLayerSize()
            val mask = Nd4j.zeros(1, getLabelSize())
            mask[all(), interval(beginIdx, till)].assign(1)
            styles.add(InputSlice(mask, beginIdx, till, styleLayer))
            beginIdx = till
        }
        val mask = Nd4j.zeros(1, getLabelSize())
        mask[all(), interval(beginIdx, beginIdx + contentLayer.flattenLayerSize())].assign(1)
        val content = InputSlice(mask, beginIdx, beginIdx + contentLayer.flattenLayerSize(), contentLayer)
        return content to styles
    }

    private data class InputSlice(
        val mask: INDArray,
        val beginIdx: Long,
        val tillIdx: Long,
        val info: NeuralTransferLayerInfo,
    )
}


