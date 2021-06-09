package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content_video"
        val style = "/Users/oleg1024/Downloads/style_video"
        val out = "/Users/oleg1024/Downloads/out_video"
        
        val alpha = "10"
        val betta = "40"

        main(arrayOf(content, style, out, "100", "1.5.15.30.50", alpha, betta, "0.03"))

    }
}