#include "../src/native_input.hpp"
#include <iostream>

int main() {
    using beam::input::TensorBinding;
    unsigned failures = 0;
    const std::string manifest = R"({"tensor_files":{"a.bin":{"size_bytes":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}})";
    const std::vector<std::byte> abc{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    auto reject = [&](const char* name, auto action) {
        try { action(); ++failures; std::cerr << "FAIL accepted " << name << '\n'; }
        catch (const std::exception&) {}
    };
    TensorBinding good(manifest, true);
    good.verify("a.bin", abc); good.finish();
    reject("changed byte", [&] { TensorBinding t(manifest, true); auto bad = abc; bad[2] = std::byte{'d'}; t.verify("a.bin", bad); });
    reject("changed size", [&] { TensorBinding t(manifest, true); t.verify("a.bin", {}); });
    reject("unlisted file", [&] { TensorBinding t(manifest, true); t.verify("b.bin", abc); });
    reject("missing loaded file", [&] { TensorBinding t(manifest, true); t.finish(); });
    reject("missing seal", [&] { TensorBinding t("{}", true); });
    reject("empty seal", [&] { TensorBinding t("{\"tensor_files\":{}}", true); });
    reject("malformed hash", [&] { TensorBinding t("{\"tensor_files\":{\"a\":{\"size_bytes\":3,\"sha256\":\"bad\"}}}", true); });
    const std::string empty = R"({"tensor_files":{"empty":{"size_bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}})";
    TensorBinding empty_test(empty, true); empty_test.verify("empty", {}); empty_test.finish();
    TensorBinding legacy("{}", false); legacy.verify("old", abc); legacy.finish();
    std::cout << "tensor_binding failures=" << failures << '\n';
    return failures ? 1 : 0;
}
