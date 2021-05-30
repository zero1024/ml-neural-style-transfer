package poa.ml.image.generator

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex
import org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex
import org.deeplearning4j.nn.layers.AbstractLayer
import org.deeplearning4j.nn.layers.FrozenLayerWithBackprop
import org.deeplearning4j.nn.modelimport.keras.KerasModel
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.learning.NoOpUpdater
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.NoOp
import org.nd4j.linalg.lossfunctions.impl.LossL2

class VGG19NeuralTransferModel2(
    h5Path: String,
    width: Long,
    height: Long,
    nChannels: Long,
) {

    private var inputGradient: INDArray? = null

    var vgg19: ComputationGraph? = null

    init {
        val modelHdf5Filename = ClassPathResource(h5Path).file.path

        val kerasModel = KerasModel().modelBuilder().modelHdf5Filename(modelHdf5Filename)
            .enforceTrainingConfig(true).buildModel()


        vgg19 = ComputationGraph(kerasModel.getComputationGraphConfiguration())
        vgg19!!.init()

//        for ((idx, vertex) in vgg19!!.vertices.withIndex()) {
//            if (vertex is PreprocessorVertex) {
//                val field = vertex::class.java.getDeclaredField("preProcessor")
//                field.isAccessible = true
//                val preProcessor = field.get(vertex) as CnnToFeedForwardPreProcessor
//                preProcessor.format = CNN2DFormat.NCHW
//            }
//        }

        vgg19 = KerasModelUtils.copyWeightsToModel(vgg19, kerasModel.layers) as ComputationGraph
        vgg19 = kerasModel.computationGraph


        vgg19 = TransferLearning.GraphBuilder(vgg19)
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
            .build()


        for (layer in vgg19!!.layers) {
            if (layer is org.deeplearning4j.nn.layers.normalization.BatchNormalization) {
                layer.layerConf().cnn2DFormat = CNN2DFormat.NCHW
            }
            if (layer is org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) {
                layer.layerConf().cnn2dDataFormat = CNN2DFormat.NCHW
            }
            if (layer is AbstractLayer<*>) {
                layer.layerConf().iDropout = null
            }
        }

        for ((idx, vertex) in vgg19!!.vertices.withIndex()) {
            if (vertex is PreprocessorVertex) {
                val field = vertex::class.java.getDeclaredField("preProcessor")
                field.isAccessible = true
                val preProcessor = field.get(vertex) as CnnToFeedForwardPreProcessor
                preProcessor.format = CNN2DFormat.NCHW
            }
        }


    }

    val loss2 = LossL2()

    private fun score(input: INDArray, labels: INDArray): Double {
        val label2d = labels2d(labels)
        return loss2.computeScore(input, label2d, ActivationIdentity(), Nd4j.ones(*label2d.shape()), true)
    }

    private fun labels2d(labels: INDArray): INDArray {
        val mb = labels.size(0)
        val labelsAsList = labels.shape().toList()
        return labels.reshape(mb, labelsAsList.subList(0, labelsAsList.size).reduce { i1, i2 -> i1 * i2 })
    }

    private fun gradient(input: INDArray, labels: INDArray): INDArray {
        val label2d = labels2d(labels)
        return loss2.computeGradient(input, label2d, ActivationIdentity(), Nd4j.ones(*label2d.shape()))
    }


    fun feedForward(img: INDArray): INDArray {
        return vgg19!!.outputSingle(img)
    }

}
