use candle_core::pickle::PthTensors;
fn main() {
    let pth = PthTensors::new("codegen_weights/pytorch_model.bin", None).unwrap();
    let infos = pth.tensor_infos();
    let mut names: Vec<&String> = infos.keys().collect();
    names.sort();
    for name in &names {
        let info = &infos[*name];
        println!("{} {:?} {:?}", name, info.layout, info.dtype);
    }
}
