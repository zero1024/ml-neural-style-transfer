package poa.ml.image.generator

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.inputs.InputType
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
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.learning.NoOpUpdater
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.NoOp
import org.nd4j.linalg.lossfunctions.impl.LossL2

class VGG19NeuralTransferModel(
    h5Path: String,
    width: Long,
    height: Long,
    nChannels: Long,
) {

    private var inputGradient: INDArray? = null

    val vgg19: ComputationGraph

    init {
        val modelHdf5Filename = ClassPathResource(h5Path).file.path
        val pretrained = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename)


        vgg19 = TransferLearning.GraphBuilder(pretrained)
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
            .addLayer(
                "outputs",
                CustomOutputLayerConf({ i, l -> score(i, l) }, { i, l -> gradient(i, l) }),
                "block5_conv4"
            )
            .setOutputs("outputs")
            .build()

        vgg19.addListeners(ScoreIterationListener(1))

        for (layer in vgg19.layers) {
            if (layer is org.deeplearning4j.nn.layers.normalization.BatchNormalization) {
                layer.layerConf().cnn2DFormat = CNN2DFormat.NCHW
            }
            if (layer is AbstractLayer<*>) {
                layer.layerConf().iDropout = null
            }
        }

        for ((idx, vertex) in vgg19.vertices.withIndex()) {
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
    }

    val loss2 = LossL2()

    private fun score(input: INDArray, labels: INDArray): Double {
        val label2d = labels2d(labels)
        return loss2.computeScore(input, label2d, ActivationIdentity(), Nd4j.ones(*label2d.shape()), true)
            .div(input.size(1) * 4)
    }

    private fun labels2d(labels: INDArray): INDArray {
        val mb = labels.size(0)
        val labelsAsList = labels.shape().toList()
        return labels.reshape(mb, labelsAsList.subList(0, labelsAsList.size).reduce { i1, i2 -> i1 * i2 })
    }

    private fun gradient(input: INDArray, labels: INDArray): INDArray {
        val label2d = labels2d(labels)
        return loss2.computeGradient(input, label2d, ActivationIdentity(), Nd4j.ones(*label2d.shape()))
            .div(input.size(1) * 4)
    }

    fun getInputGradient(img: INDArray, label: INDArray): INDArray {
        vgg19.fit(arrayOf(img), arrayOf(label))
        return inputGradient!!
    }

    fun feedForward(img: INDArray): Map<String, INDArray> {
        return vgg19.feedForward(img, false)
    }

}
