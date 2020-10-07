mod abs_model;
mod dataset;
mod mlp;
use clap::Clap;

#[derive(Clap)]
#[clap(version = "1.0", author = "CNLHC <buaa_cnlhc@buaa.edu.cn>")]
struct Opts {
    #[clap(short, long)]
    train: bool,
    #[clap(short, long)]
    load: bool,
}

use abs_model::MNISTModel;

fn main() {
    let opts: Opts = Opts::parse();

    let mut model: mlp::MNISTMlp = abs_model::MNISTModel::new();
    if opts.load {
        model.load_model();
    }
    if opts.train {
        model.train();
    }

    let acc = model.validate();
    println!("accuracy: {:.2}%", acc * 100.0);
}
