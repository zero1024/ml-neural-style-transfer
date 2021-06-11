package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content_portrait2"
        val style = "/Users/oleg1024/Downloads/style_portrait2"
        val out = "/Users/oleg1024/Downloads/out_portrait2"

        val alpha = "10"
        val betta = "1,10,40,100"

        main(arrayOf(content, style, out, "5,15,50,100,200", alpha, betta, "0.03"))

    }
}