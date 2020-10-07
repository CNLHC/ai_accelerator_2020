extern crate autograd as ag;
use crate::abs_model::{MNISTModel, NdArray};
use crate::dataset::load;
use ag::ndarray_ext as array;
use ag::optimizers::adam;
use ag::rand::seq::SliceRandom;
use ag::tensor::Variable;
use ag::Graph;
use ndarray::s;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::sync::{Arc, RwLock};

type Tensor<'graph> = ag::Tensor<'graph, f32>;
fn inputs(g: &Graph<f32>) -> (Tensor, Tensor) {
    let x = g.placeholder(&[-1, 28 * 28]);
    let y = g.placeholder(&[-1, 1]);
    (x, y)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

pub struct MNISTMlp {
    x_train: NdArray,
    y_train: NdArray,
    x_test: NdArray,
    y_test: NdArray,
    hidden1: Arc<RwLock<NdArray>>,
    hidden2: Arc<RwLock<NdArray>>,
    biased: Arc<RwLock<NdArray>>,
}

impl MNISTModel for MNISTMlp {
    fn new() -> MNISTMlp {
        let ((x_train, y_train), (x_test, y_test)) = load();
        let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
        let model = MNISTMlp {
            x_train: x_train,
            y_train: y_train,
            x_test: x_test,
            y_test: y_test,
            hidden1: array::into_shared(rng.glorot_uniform(&[28 * 28, 128])),
            hidden2: array::into_shared(rng.glorot_uniform(&[128, 10])),
            biased: array::into_shared(array::zeros(&[1, 10])),
        };
        return model;
    }

    fn load_model(&mut self) -> Result<(), std::io::Error> {
        self.hidden1 =
            array::into_shared(NdArray::read_npy(File::open("./hidden1.npy")?).expect("err"));
        self.hidden2 =
            array::into_shared(NdArray::read_npy(File::open("./hidden2.npy")?).expect("err"));
        self.biased =
            array::into_shared(NdArray::read_npy(File::open("./biased.npy")?).expect("err"));
        Ok(())
    }

    fn train(&self) {
        let adam_state = adam::AdamState::new(&[&self.hidden1, &self.hidden2, &self.biased]);
        let max_epoch = 5;
        let batch_size = 50isize;
        let num_samples = self.x_train.shape()[0];
        let num_batches = num_samples / batch_size as usize;

        for epoch in 0..max_epoch {
            ag::with(|g| {
                let h1 = g.variable(self.hidden1.clone());
                let h2 = g.variable(self.hidden2.clone());
                let b = g.variable(self.biased.clone());
                let (x, y) = inputs(g);
                let z = g.matmul(g.relu(g.matmul(x, h1)), h2) + b;
                let loss = g.sparse_softmax_cross_entropy(z, &y);
                let mean_loss = g.reduce_mean(loss, &[0], false);
                let grads = &g.grad(&[&mean_loss], &[h1, h2, b]);
                let update_ops: &[Tensor] =
                    &adam::Adam::default().compute_updates(&[h1, h2, b], grads, &adam_state, g);

                for i in get_permutation(num_batches) {
                    let i = i as isize * batch_size;
                    let x_batch = self.x_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    let y_batch = self.y_train.slice(s![i..i + batch_size, ..]).into_dyn();
                    g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
                }
                println!("finish epoch {}", epoch);
            });
        }

        match self.hidden1.try_read() {
            Ok(arr) => (*arr)
                .write_npy(File::create("hidden1.npy").expect("err"))
                .expect("err"),
            Err(_) => {}
        }
        match self.hidden2.try_read() {
            Ok(arr) => (*arr)
                .write_npy(File::create("hidden2.npy").expect("err"))
                .expect("err"),
            Err(_) => {}
        }

        match self.biased.try_read() {
            Ok(arr) => (*arr)
                .write_npy(File::create("biased.npy").expect("err"))
                .expect("err"),
            Err(_) => (),
        }
    }

    fn validate(&self) -> f64 {
        let mut res: f64 = 0.0;
        ag::with(|g| {
            let h1 = g.variable(self.hidden1.clone());
            let h2 = g.variable(self.hidden2.clone());
            let b = g.variable(self.biased.clone());
            let (x, y) = inputs(g);
            let z = g.matmul(g.relu(g.matmul(x, h1)), h2) + b;
            let predictions = g.argmax(z, -1, true);
            let accuracy = g.reduce_mean(&g.equal(predictions, &y), &[0, 1], false);
            res = accuracy
                .eval(&[x.given(self.x_test.view()), y.given(self.y_test.view())])
                .expect("e")
                .scalar_sum() as f64;
        });
        return res;
    }
}
