# Rust MNIST

## How To Run

1. make sure you have `rust2018` and `cargo` installed.
```
>cargo --version
 cargo 1.46.0 (149022b1d 2020-07-17)
>rustc --version
 rustc 1.46.0 (04488afe3 2020-08-24)
```

2. Download the mnist data from [here](http://yann.lecun.com/exdb/mnist/), and put them in `./data/` directory.(Don't forget to unzip them using `gzip -d`)
```
data
├── t10k-images-idx3-ubyte
├── t10k-labels-idx1-ubyte
├── train-images-idx3-ubyte
└── train-labels-idx1-ubyte
```

3. build the project
 ```
 cargo build
 ```

4. run with training
 ```
 cargo run -- --train
 ```

5. run with pretrained parameters
 ```
 cargo run -- --load
 ```


