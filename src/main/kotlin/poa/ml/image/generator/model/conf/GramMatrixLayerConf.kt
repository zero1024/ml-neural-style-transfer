package poa.ml.image.generator.model.conf

import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import poa.ml.image.generator.convGramMatrix

class GramMatrixLayerConf(private val mb: Long = 1) : SameDiffLambdaLayer() {

    override fun defineLayer(sameDiff: SameDiff, input: SDVariable) = sameDiff.convGramMatrix(mb, input)

    override fun getOutputType(layerIndex: Int, inputType: InputType): InputType {
        inputType as InputType.InputTypeConvolutional
        return InputType.feedForward(inputType.channels * inputType.channels)
    }
}