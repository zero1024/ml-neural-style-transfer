package poa.ml.image.generator.model.layer

abstract class NeuralTransferLayerInfo(
    val name: String,
    val height: Long,
    val width: Long,
    val nChannels: Long,
) {
    abstract fun flattenLayerName(): String
    abstract fun flattenLayerSize(): Long

}