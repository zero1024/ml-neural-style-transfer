package poa.ml.image.generator.model

import org.nd4j.linalg.api.ndarray.INDArray

interface NeuralTransferModel {

    fun rescaleBack(img: INDArray, width: Int, height: Int): INDArray

    fun scaleForModel(img: INDArray): INDArray

    fun inputGradient(img: INDArray, label: INDArray): INDArray

    fun toLabel(contentImg: INDArray, styleImg: INDArray): INDArray

    fun score(): Double

}


