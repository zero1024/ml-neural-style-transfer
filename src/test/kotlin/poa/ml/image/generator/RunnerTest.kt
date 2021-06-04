package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content"
        val style = "/Users/oleg1024/Downloads/style"
        val out = "/Users/oleg1024/Downloads/out"

        main(arrayOf(content, style, out))

    }
}