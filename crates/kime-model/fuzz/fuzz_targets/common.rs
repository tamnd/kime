// Shared by the targets: read every byte of every tensor the parser accepted, so an offset that
// escaped the checks shows up as a panic rather than passing quietly.
pub fn touch(t: &kime_model::Tensors) {
    let mut sum = 0u64;
    for i in 0..t.entries().len() {
        let v = t.view(i);
        sum = sum.wrapping_add(v.bytes.iter().map(|&b| u64::from(b)).sum::<u64>());
        if v.bytes.len() <= 4096 {
            std::hint::black_box(v.to_f32());
        }
    }
    std::hint::black_box(sum);
}
