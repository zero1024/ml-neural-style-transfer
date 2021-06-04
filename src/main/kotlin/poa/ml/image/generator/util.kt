package poa.ml.image.generator

import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Size
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Toolkit
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.SimpleFileVisitor
import java.nio.file.attribute.BasicFileAttributes
import javax.imageio.ImageIO
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.WindowConstants

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

fun BufferedImage.resize(height: Int, width: Int): BufferedImage {
    val resizedImage = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics2D = resizedImage.createGraphics()
    graphics2D.drawImage(this, 0, 0, width, height, null)
    graphics2D.dispose()
    return resizedImage
}

fun showImage(image: BufferedImage, lambda: (JFrame) -> Unit = {}) {
    showJLabel(JLabel(ImageIcon(image)), lambda)
}

fun showJLabel(jLabel: JLabel, lambda: (JFrame) -> Unit = {}) {
    val frame = JFrame()
    val dim: Dimension = Toolkit.getDefaultToolkit().screenSize
    frame.contentPane.layout = FlowLayout()
    frame.contentPane.add(jLabel)
    frame.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
    lambda(frame)
    frame.pack()
    frame.setLocation(dim.width / 2 - frame.size.width / 2, dim.height / 2 - frame.size.height / 2)
    frame.isVisible = true
}

fun walkFileTree(path: String, lambda: (File) -> Unit) {
    Files.walkFileTree(Path.of(path), object : SimpleFileVisitor<Path>() {
        override fun visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult {
            if (file.toFile().name.isImage()) {
                lambda(file.toFile())
            }
            return FileVisitResult.CONTINUE
        }

        private fun String.isImage() = endsWith(".jpg") || endsWith(".png") || endsWith(".jpeg")
    })
}

fun NativeImageLoader.resize(img: INDArray, width: Int, height: Int): INDArray {
    val mat = asMat(img)
    opencv_imgproc.resize(mat, mat, Size(width, height))
    return asMatrix(mat)
}