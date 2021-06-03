package poa.ml.image.generator

import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray

fun SameDiff.convGramMatrix(mb: Long = 0, input: SDVariable): SDVariable {
    val nChannels = input.shape[1]
    val height = input.shape[2]
    val width = input.shape[3]
    val gramMatrices = (0 until mb).map {
        val channelsInput = input[SDIndex.point(it)].reshape(nChannels, height * width)
        val gram = this.mmul(channelsInput, this.transpose(channelsInput))
        gram.reshape(nChannels * nChannels)
    }.toTypedArray()
    return this.stack(0, *gramMatrices)
}

fun SameDiff.applyMask(sdVariable: SDVariable, mask: INDArray) = sdVariable.mul(`var`(mask))