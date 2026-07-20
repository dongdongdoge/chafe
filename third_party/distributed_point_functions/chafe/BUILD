load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "google_dpf_write_benchmark",
    srcs = ["google_dpf_write_benchmark.cc"],
    deps = [
        "//dpf:distributed_point_function",
        "//dpf:tuple",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "google_dpf_loopback_e2e_benchmark",
    srcs = ["google_dpf_loopback_e2e_benchmark.cc"],
    deps = [
        "//dpf:distributed_point_function",
        "//pir/prng:aes_128_ctr_seeded_prng",
        "@boringssl//:crypto",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
