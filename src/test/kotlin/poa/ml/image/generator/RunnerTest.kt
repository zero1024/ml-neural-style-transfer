package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content2"
        val style = "/Users/oleg1024/Downloads/style"
        val out = "/Users/oleg1024/Downloads/out2"

        val alpha = "10"
        val betta = "40"

        main(arrayOf(content, style, out, "100", "1.3.5.10.20.30.50", alpha, betta))

    }
}