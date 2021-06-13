package poa.ml.neural.style.transfer

import org.bytedeco.opencv.opencv_core.Mat
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory
import poa.ml.neural.style.transfer.model.DarknetNeuralTransferModel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.min

private val logger = LoggerFactory.getLogger("main")

fun main(args: Array<String>) {

    val contentDir = args[0]
    val styleDir = args[1]
    val outDir = args[2]
    val iterations = args.getOrElse(3) { "50" }.split(",").map { it.toInt() }
    val alpha = args.getOrElse(4) { "10.0" }.toDouble()
    val bettaList = args.getOrElse(5) { "40.0" }.split(",").map { it.toDouble() }
    val lr = args.getOrElse(6) { "0.03" }.toDouble()
    val styleWeights =
        args.getOrElse(7) { "0.2,0.2,0.2,0.2,0.2" }.split(";")
            .map { a -> a.split(",").map { n -> n.toDouble() } }

    //todo validation
    assert(styleWeights.isNotEmpty())
    styleWeights.forEach { assert(it.size == 5) }

    logger.info("Starting neural style transferring with parameters: ${args.toList()}")

    for (betta in bettaList) {
        for (styleWeight in styleWeights) {
            logger.info("Starting neural style transferring with betta = $betta and style weights = $styleWeight")
            run(alpha, betta, styleDir, contentDir, lr, iterations, outDir, styleWeight)
        }
    }
}

private fun run(
    alpha: Double,
    betta: Double,
    styleDir: String,
    contentDir: String,
    lr: Double,
    iterations: List<Int>,
    outDir: String,
    styleWeight: List<Double>,
) {
    val imageLoader = NativeImageLoader()
    val model = DarknetNeuralTransferModel(alpha = alpha, betta = betta, styleWeighs = styleWeight.toDoubleArray())

    logger.info("Loading styles...")
    val styles = mutableMapOf<String, INDArray>()
    walkFileTree(styleDir) { f ->
        val style = imageLoader.asMatrix(f)
        val resizedStyle = model.scaleForModel(style)
        styles[f.nameWithoutExtension] = resizedStyle
    }
    logger.info("Styles ${styles.keys} are loaded.")

    logger.info("Walking through content directory...")
    walkFileTree(contentDir) { f ->
        logger.info("Found [${f.name}]")
        val content = imageLoader.asMatrix(f)
        val origHeight = content.size(2).toInt()
        val origWidth = content.size(3).toInt()
        val (height, width) = chooseReasonableSize(origHeight, origWidth)
        val resizedContent = model.scaleForModel(content)
        val contentName = f.nameWithoutExtension

        for ((styleName, style) in styles) {

            val allFilesExist = iterations
                .map { outFileName(outDir, contentName, styleName, it, betta, styleWeight) }
                .all { Files.exists(Path.of(it)) }
            if (allFilesExist) {
                logger.info("File [$contentName] with style [${styleName}] already exist. Moving forward. ")
                continue
            }

            logger.info("Applying style [${styleName}] to [$contentName]")
            val label = model.toLabel(resizedContent, style)
            val updater = adamUpdater(lr, resizedContent.shape())

            var img = resizedContent.add(0.0)

            for (i in 0..iterations.last()) {
                val res = model.inputGradient(img, label)
                logger.info("Iteration $i. Score - ${model.score()}")
                updater.applyUpdater(res, i, 0)
                img = img.sub(res)
                if (iterations.contains(i)) {
                    val outFilePath = outFileName(outDir, contentName, styleName, i, betta, styleWeight)
                    val newImg = model.rescaleBack(img, width, height)
                    logger.info("Saving [${outFilePath}]...")
                    saveImage(outFilePath, imageLoader.asMat(newImg))
                    logger.info("Saving done.")
                }
            }
        }
    }
}

private fun outFileName(
    outDir: String,
    fileName: String,
    styleName: String,
    i: Int,
    betta: Double,
    styleWeight: List<Double>,
): String {
    return "${outDir}/${fileName}_${styleName}_iter_${i}_betta_${betta}_sw_${styleWeight}.jpg"
}

private fun adamUpdater(lr: Double, shape: LongArray): GradientUpdater<*> {
    return Adam(lr).instantiate(mapOf(
        "M" to Nd4j.create(*shape),
        "V" to Nd4j.create(*shape)),
        true)
}

private fun chooseReasonableSize(height: Int, width: Int): Pair<Int, Int> {
    val minSize = min(height, width)
    val reasonableMax = min(minSize, 500)
    val scale: Double = minSize.toDouble() / reasonableMax
    return (height.toDouble() / scale).toInt() to (width.toDouble() / scale).toInt()
}

private fun saveImage(path: String, img: Mat) {
    org.bytedeco.opencv.global.opencv_imgcodecs.imwrite(path, img)
}
