use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let data: [u32; 3] = [1u32, 2, 3];
    let _tensor = Tensor::new(&data, &Device::Cpu)?;

    let nested_data: [[[u32; 3]; 3]; 3] = [
        [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]],
    ];
    let tensor = Tensor::new(&nested_data, &Device::Cpu)?;

    // Tensor: Tensor[dims 3, 3, 3; u32]
    println!("Tensor: {:?}", tensor);

    // Tensor shape: [3, 3, 3]
    println!("Tensor shape: {:?}", tensor.shape());
    // Tensor shape dims: [3, 3, 3]
    println!("Tensor shape dims: {:?}", tensor.shape().dims());
    // Tensor shape dims2: Err(unexpected rank, expected: 2, got: 3 ([3, 3, 3]))
    println!("Tensor shape dims2: {:?}", tensor.shape().dims2());
    // Tensor shape dims3: Ok((3, 3, 3))
    println!("Tensor shape dims3: {:?}", tensor.shape().dims3());

    // Tensor dims: [3, 3, 3]
    println!("Tensor dims: {:?}", tensor.dims());
    // Tensor dims2: Err(unexpected rank, expected: 2, got: 3 ([3, 3, 3]))
    println!("Tensor dims2: {:?}", tensor.dims2());
    // Tensor dims3: Ok((3, 3, 3))
    println!("Tensor dims3: {:?}", tensor.dims3());
    // Tensor dims4: Err(unexpected rank, expected: 4, got: 3 ([3, 3, 3]))
    println!("Tensor dims4: {:?}", tensor.dims4());
    // Tensor dims5: Err(unexpected rank, expected: 5, got: 3 ([3, 3, 3]))
    println!("Tensor dims5: {:?}", tensor.dims5());

    Ok(())
}
