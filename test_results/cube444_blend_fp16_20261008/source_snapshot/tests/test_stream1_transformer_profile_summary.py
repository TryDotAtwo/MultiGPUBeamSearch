import sqlite3
import unittest

from tools.stream1_transformer_profile_summary import KernelRow, read_nsys_kernel_rows, summarize_kernels


class ProfileSummaryTests(unittest.TestCase):
    def test_aggregation_preserves_time_and_classifies_unknown(self):
        rows = [
            KernelRow("cutlass::Kernel2", 3_000),
            KernelRow("attention_kernel_batched_impl", 2_000),
            KernelRow("stream1_transformer_layernorm256_copy_kernel", 1_000),
            KernelRow("unexpected_kernel", 500),
        ]
        result = summarize_kernels(rows)
        self.assertEqual(sum(item.total_ns for item in result), 6_500)
        self.assertEqual(result[0].family, "gemm_fused")
        self.assertEqual(result[-1].family, "other")

    def test_rejects_negative_duration(self):
        with self.assertRaisesRegex(ValueError, "duration_ns"):
            summarize_kernels([KernelRow("kernel", -1)])

    def test_rejects_empty_profile(self):
        with self.assertRaisesRegex(ValueError, "no kernel rows"):
            summarize_kernels([])

    def test_reads_nsys_kernel_rows_and_resolves_demangled_names(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, end INTEGER NOT NULL, demangledName INTEGER NOT NULL)")
        connection.execute("INSERT INTO StringIds VALUES (7, 'cutlass::Kernel2')")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 160, 7)")
        self.assertEqual(read_nsys_kernel_rows(connection), [KernelRow("cutlass::Kernel2", 60)])

    def test_rejects_missing_nsys_kernel_table(self):
        connection = sqlite3.connect(":memory:")
        with self.assertRaisesRegex(ValueError, "CUPTI_ACTIVITY_KIND_KERNEL"):
            read_nsys_kernel_rows(connection)

    def test_attention_name_with_cutlass_template_is_not_plain_gemm(self):
        rows = [KernelRow("attention_kernel_batched_impl<AttentionKernel<cutlass::half_t>>", 100)]
        self.assertEqual(summarize_kernels(rows)[0].family, "attention")


if __name__ == "__main__":
    unittest.main()
