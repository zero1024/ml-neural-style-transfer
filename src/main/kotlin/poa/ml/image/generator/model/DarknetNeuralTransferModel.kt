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
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.zoo.model.Darknet19
import org.nd4j.autodiff.loss.LossReduce
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
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

class DarknetNeuralTransferModel(
    mb: Long = 1,
    styleWeighs: DoubleArray = doubleArrayOf(0.2, 0.2, 0.2, 0.2, 0.2),
    private val alpha: Double = 10.0,
    private val betta: Double = 10.0,
) : NeuralTransferModel {


    private val imageLoader = NativeImageLoader()
    private var inputGradient: INDArray? = null
    private val model: ComputationGraph

    private val width = 448L
    private val height = 448L
    private val nChannels = 3L

    private val contentLayer = ContentNeuralTransferLayerInfo("conv2d_18", 14, 14, 1024)
    private val stylesLayers = listOf(
        StyleNeuralTransferLayerInfo("conv2d_2", 224, 224, 64, weight = styleWeighs[0]),
        StyleNeuralTransferLayerInfo("conv2d_5", 112, 112, 128, weight = styleWeighs[1]),
        StyleNeuralTransferLayerInfo("conv2d_8", 56, 56, 256, weight = styleWeighs[2]),
        StyleNeuralTransferLayerInfo("conv2d_11", 28, 28, 512, weight = styleWeighs[3]),
        StyleNeuralTransferLayerInfo("conv2d_16", 14, 14, 1024, weight = styleWeighs[4])
    )

    init {

        val pretrained =
            Darknet19.builder().inputShape(intArrayOf(nChannels.toInt(), width.toInt(), height.toInt())).build()
                .initPretrained() as ComputationGraph

        val preBuild = TransferLearning.GraphBuilder(pretrained)
            .fineTuneConfiguration(FineTuneConfiguration.Builder().build())
            .setInputTypes(InputType.convolutional(
                height,
                width,
                nChannels,
                CNN2DFormat.NCHW
            ))
            .removeVertexAndConnections("globalpooling")
            .removeVertexAndConnections("softmax")
            .removeVertexAndConnections("loss")
            .removeVertexAndConnections("conv2d_19")
            .removeVertexAndConnections("leaky_re_lu_18")
            .removeVertexAndConnections("batch_normalization_18")

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
                field.set(vertex, object : FrozenLayerWithBackprop(vertex.layer) {

                    override fun activate(training: Boolean, workspaceMgr: LayerWorkspaceMgr?): INDArray {
                        return underlying.activate(true, workspaceMgr)
                    }

                    override fun activate(
                        input: INDArray?,
                        training: Boolean,
                        workspaceMgr: LayerWorkspaceMgr?,
                    ): INDArray {
                        return underlying.activate(input, true, workspaceMgr)
                    }

                })
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

        val tmp = res[all(), point(0), all(), all()].add(0.0)
        res[all(), point(0), all(), all()].assign(res[all(), point(2), all(), all()])
        res[all(), point(2), all(), all()].assign(tmp)

        return res
    }

    override fun rescaleBack(img: INDArray, width: Int, height: Int): INDArray {
        val res = imageLoader.resize(img, width, height)
        res[all(), point(0), all(), all()].muli(255)
        res[all(), point(1), all(), all()].muli(255)
        res[all(), point(2), all(), all()].muli(255)
        res.replaceWhere(Nd4j.create(*res.shape()).assign(255), GreaterThan(255))
        res.replaceWhere(Nd4j.create(*res.shape()).assign(0), LessThan(0))

        val tmp = res[all(), point(0), all(), all()].add(0.0)
        res[all(), point(0), all(), all()].assign(res[all(), point(2), all(), all()])
        res[all(), point(2), all(), all()].assign(tmp)

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

        val styleActivations = model.feedForward(styleImg, true)
        for (styleSlice in styleSlices) {
            val styleLabel = styleActivations[styleSlice.info.flattenLayerName()]
            label2D[all(), interval(styleSlice.beginIdx, styleSlice.tillIdx)]
                .assign(styleLabel)
        }

        val contentActivations = model.feedForward(contentImg, true)
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


