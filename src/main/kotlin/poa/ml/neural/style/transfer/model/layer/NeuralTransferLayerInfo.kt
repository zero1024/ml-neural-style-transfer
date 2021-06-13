package poa.ml.neural.style.transfer.model.layer

abstract class NeuralTransferLayerInfo(
    val name: String,
    val height: Long,
    val width: Long,
    val nChannels: Long,
) {
    abstract fun flattenLayerName(): String
    abstract fun flattenLayerSize(): Long

}