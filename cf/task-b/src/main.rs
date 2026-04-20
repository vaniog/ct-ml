use std::{
    cell::RefCell,
    fmt::Debug,
    io::{self, BufWriter, Read, Write},
    iter::Sum,
    ops::{Add, Mul},
    rc::Rc,
    vec,
};

#[derive(Clone)]
struct Matrix {
    w: usize,
    h: usize,
    m: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(v: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            w: v.len(),
            h: v[0].len(),
            m: v,
        }
    }

    fn apply(&self, f: impl Fn(f64) -> f64) -> Matrix {
        let mut m_new = vec![vec![0.; self.h]; self.w];
        for x in 0..self.w {
            for y in 0..self.h {
                m_new[x][y] = f(self.m[x][y])
            }
        }
        Matrix {
            m: m_new,
            w: self.w,
            h: self.h,
        }
    }

    fn tanh(&self) -> Matrix {
        self.apply(f64::tanh)
    }

    fn relu(&self, a: f64) -> Matrix {
        self.apply(|x| if x >= 0. { x } else { a * x })
    }

    fn had(&self, rhs: Matrix) -> Matrix {
        assert!(self.w == rhs.w && self.h == rhs.h);
        let mut m_new = vec![vec![0.; self.h]; self.w];
        for x in 0..self.w {
            for y in 0..self.h {
                m_new[x][y] = self.m[x][y] * rhs.m[x][y]
            }
        }
        Matrix {
            m: m_new,
            w: self.w,
            h: self.h,
        }
    }

    fn had_div(&self, rhs: Matrix) -> Matrix {
        assert!(self.w == rhs.w && self.h == rhs.h);

        let mut m_new = vec![vec![0.; self.h]; self.w];
        for x in 0..self.w {
            for y in 0..self.h {
                m_new[x][y] = self.m[x][y] / rhs.m[x][y]
            }
        }
        Matrix {
            m: m_new,
            w: self.w,
            h: self.h,
        }
    }

    fn t(&self) -> Matrix {
        let mut m_new = vec![vec![0.; self.w]; self.h];

        for x in 0..self.w {
            for y in 0..self.h {
                m_new[y][x] = self.m[x][y]
            }
        }
        Matrix {
            m: m_new,
            w: self.h,
            h: self.w,
        }
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.h {
            for x in 0..self.w {
                write!(f, "{:.10} ", self.m[x][y])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[\n")?;
        for y in 0..self.h {
            write!(f, "[")?;
            for x in 0..self.w {
                write!(f, "{} ", self.m[x][y])?;
            }
            write!(f, "]\n")?;
        }
        write!(f, "]\n")?;
        Ok(())
    }
}

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(self.w == rhs.w && self.h == rhs.h);

        let mut m_new = vec![vec![0.; self.h]; self.w];

        for x in 0..self.w {
            for y in 0..self.h {
                m_new[x][y] = self.m[x][y] + rhs.m[x][y]
            }
        }

        Matrix { m: m_new, ..self }
    }
}

impl Sum for Matrix {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut m = iter.next().unwrap();
        for i in iter {
            m = m + i;
        }
        m
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.w == rhs.h);
        let mut m_new = vec![vec![0.; self.h]; rhs.w];

        for x in 0..rhs.w {
            for y in 0..self.h {
                let mut mxy = 0.;
                for i in 0..self.w {
                    mxy = mxy + self.m[i][y] * rhs.m[x][i];
                }
                m_new[x][y] = mxy;
            }
        }

        Matrix {
            w: rhs.w,
            h: self.h,
            m: m_new,
        }
    }
}

trait Block {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>);
}

struct Node {
    f: Box<dyn Block>,
    args: Vec<Rc<RefCell<Node>>>,
    backward_f: Option<Box<dyn Fn(Matrix) -> Vec<Matrix>>>,
    grad: Option<Matrix>,
}

impl Node {
    fn new(f: Box<dyn Block>, args: Vec<Rc<RefCell<Node>>>) -> Node {
        Node {
            f,
            args,
            backward_f: None,
            grad: None,
        }
    }

    fn forward(&mut self) -> Matrix {
        let args = self.args.iter().map(|n| n.borrow_mut().forward()).collect();
        let (a, backward) = self.f.forward(args);
        self.backward_f = Some(Box::new(backward));
        a
    }

    fn backward(&mut self, grad: Matrix) {
        for (i, x) in self
            .backward_f
            .as_ref()
            .expect("backward called after forward")(grad.clone())
        .iter()
        .enumerate()
        {
            self.args[i].as_ref().borrow_mut().backward(x.clone());
        }
        match self.grad.clone() {
            Some(g) => self.grad = Some(g + grad.clone()),
            None => self.grad = Some(grad),
        }
    }
}

struct SumBlock {}
impl SumBlock {
    fn new() -> Box<SumBlock> {
        Box::new(SumBlock {})
    }
}

impl Block for SumBlock {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        (
            args.clone().into_iter().sum(),
            Box::new(move |grad| (vec![grad.clone(); args.len()])),
        )
    }
}

struct HadBlock {}
impl HadBlock {
    fn new() -> Box<HadBlock> {
        Box::new(HadBlock {})
    }
}

impl Block for HadBlock {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        let mut m = args[0].clone();
        for i in 1..args.len() {
            m = m.had(args[i].clone())
        }

        (
            m.clone(),
            Box::new(move |grad| {
                let mut v = Vec::<Matrix>::new();
                for i in 0..args.len() {
                    v.push(m.had_div(args[i].clone()).had(grad.clone()));
                }
                v
            }),
        )
    }
}

struct MultiplyBlock {}

impl MultiplyBlock {
    fn new() -> Box<MultiplyBlock> {
        Box::new(MultiplyBlock {})
    }
}
impl Block for MultiplyBlock {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        assert!(args.len() == 2);
        return (
            args[0].clone() * args[1].clone(),
            Box::new(move |grad| {
                vec![
                    grad.clone() * args[1].clone().t(),
                    args[0].clone().t() * grad.clone(),
                ]
            }),
        );
    }
}

struct MatrixBlock {
    m: Rc<RefCell<Matrix>>,
}

impl MatrixBlock {
    fn new(m: Rc<RefCell<Matrix>>) -> Box<MatrixBlock> {
        Box::new(MatrixBlock { m: Rc::clone(&m) })
    }
}

impl Block for MatrixBlock {
    fn forward(&self, _: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        (self.m.as_ref().borrow().clone(), Box::new(|_| vec![]))
    }
}

struct ReLUBlock {
    a: f64,
}

impl ReLUBlock {
    fn new(a: f64) -> Box<ReLUBlock> {
        Box::new(ReLUBlock { a })
    }
}

impl Block for ReLUBlock {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        assert!(args.len() == 1);
        let a = self.a;
        (
            args[0].clone().relu(self.a),
            Box::new(move |grad| {
                let m = args[0].clone();
                let mut grad_new = vec![vec![0.; grad.h]; grad.w];

                for x in 0..grad.w {
                    for y in 0..grad.h {
                        grad_new[x][y] = grad.m[x][y] * if m.m[x][y] >= 0. { 1. } else { a };
                    }
                }
                vec![Matrix {
                    m: grad_new,
                    w: m.w,
                    h: m.h,
                }]
            }),
        )
    }
}

struct TanhBlock {}

impl TanhBlock {
    fn new() -> Box<TanhBlock> {
        Box::new(TanhBlock {})
    }
}

impl Block for TanhBlock {
    fn forward(&self, args: Vec<Matrix>) -> (Matrix, Box<dyn Fn(Matrix) -> Vec<Matrix>>) {
        assert!(args.len() == 1);
        (
            args[0].clone().tanh(),
            Box::new(move |grad| {
                let m = args[0].clone();
                let mut grad_new = vec![vec![0.; grad.h]; grad.w];

                for x in 0..grad.w {
                    for y in 0..grad.h {
                        grad_new[x][y] = grad.m[x][y] * (1. - m.m[x][y].tanh().powi(2));
                    }
                }
                vec![Matrix {
                    m: grad_new,
                    w: m.w,
                    h: m.h,
                }]
            }),
        )
    }
}

#[allow(unused_macros)]
macro_rules! read {
    ($in:ident >> $out:ident as $type:ty) => {
        let next = $in.next();
        if let None = next {
            while (true) {}
        }
        let $out = next
            .expect("Unwrappable")
            .parse::<$type>()
            .expect("Parsable");
    };
}

#[allow(unused_macros)]
macro_rules! read_str {
    ($in:ident >> $out:ident) => {
        let next = $in.next();
        if let None = next {
            while (true) {}
        }
        let $out = next.expect("Unwrappable");
    };
}

fn main() -> io::Result<()> {
    let mut input_line = String::new();
    io::stdin().read_to_string(&mut input_line)?;
    println!("{}", solve(&input_line)?);
    Ok(())
}

fn solve(s: &str) -> io::Result<String> {
    let mut input = s.split_ascii_whitespace();
    let mut output = BufWriter::new(Vec::new());

    read!(input >> n as u32);
    read!(input >> m as u32);
    read!(input >> k as u32);

    let mut nodes = Vec::<Rc<RefCell<Node>>>::new();
    let mut vars = Vec::<Rc<RefCell<Matrix>>>::new();

    for _ in 0..n {
        read_str!(input >> op);
        match op {
            "var" => {
                read!(input >> h as usize);
                read!(input >> w as usize);
                let m = Rc::new(RefCell::new(Matrix::new(vec![vec![0.; h]; w])));
                vars.push(Rc::clone(&m));
                nodes.push(Rc::new(RefCell::new(Node::new(
                    MatrixBlock::new(m),
                    vec![],
                ))));
            }
            "sum" => {
                read!(input >> len as usize);
                let mut args = Vec::<Rc<RefCell<Node>>>::new();
                for _ in 0..len {
                    read!(input >> i as usize);
                    args.push(Rc::clone(&nodes[i - 1]));
                }
                nodes.push(Rc::new(RefCell::new(Node::new(SumBlock::new(), args))));
            }
            "mul" => {
                let mut args = Vec::<Rc<RefCell<Node>>>::new();
                for _ in 0..2 {
                    read!(input >> i as usize);
                    args.push(Rc::clone(&nodes[i - 1]));
                }
                nodes.push(Rc::new(RefCell::new(Node::new(MultiplyBlock::new(), args))));
            }
            "rlu" => {
                read!(input >> a as f64);
                read!(input >> i as usize);
                nodes.push(Rc::new(RefCell::new(Node::new(
                    ReLUBlock::new(1. / a),
                    vec![Rc::clone(&nodes[i - 1])],
                ))));
            }
            "tnh" => {
                read!(input >> i as usize);
                nodes.push(Rc::new(RefCell::new(Node::new(
                    TanhBlock::new(),
                    vec![Rc::clone(&nodes[i - 1])],
                ))));
            }
            "had" => {
                read!(input >> len as usize);
                let mut args = Vec::<Rc<RefCell<Node>>>::new();
                for _ in 0..len {
                    read!(input >> i as usize);
                    args.push(Rc::clone(&nodes[i - 1]));
                }
                nodes.push(Rc::new(RefCell::new(Node::new(HadBlock::new(), args))));
            }

            &_ => unreachable!(),
        };
    }

    for var in vars.iter() {
        let mut var = var.as_ref().borrow_mut();
        for y in 0..var.h {
            for x in 0..var.w {
                read!(input >> c as f64);
                var.m[x][y] = c;
            }
        }
    }

    let mut outputs = Vec::<Matrix>::new();
    let mut grads = Vec::<Matrix>::new();

    for i in 0..k {
        outputs.push(nodes[(n - k + i) as usize].as_ref().borrow_mut().forward());
        write!(output, "{}", outputs[i as usize])?;
    }

    for i in 0..k {
        let mut m = vec![vec![0.; outputs[i as usize].h]; outputs[i as usize].w];
        for y in 0..outputs[i as usize].h {
            for x in 0..outputs[i as usize].w {
                read!(input >> c as f64);
                m[x][y] = c;
            }
        }
        grads.push(Matrix::new(m));
    }

    for (i, g) in grads.iter().enumerate() {
        nodes[(n - k + i as u32) as usize]
            .as_ref()
            .borrow_mut()
            .backward(g.clone());
    }

    for i in 0..m {
        let n = nodes[i as usize].as_ref().borrow();
        match n.grad.clone() {
            Some(g) => {
                write!(output, "{}", g)?;
            }
            None => {
                let var = vars[i as usize].as_ref().borrow_mut();
                write!(output, "{}", Matrix::new(vec![vec![0.; var.h]; var.w]))?;
            }
        }
    }

    Ok(String::from_utf8(output.into_inner()?).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
        assert_eq!(
            solve(
                "
5 3 2
var 2 1
var 2 1
var 2 1
had 3 1 2 3
sum 3 1 2 2

1
2

2
3

3
4

5
5

1
1
                "
            )
            .unwrap(),
            "6 
24 

5 
8 

31 
61 

17 
42 

10 
30 

"
        );
    }

    #[test]
    fn test_2() {
        assert_eq!(
            solve(
                "
4 2 2
var 1 1
var 1 1
rlu 2 2
rlu 2 1
1
-1
1
1
            "
            )
            .unwrap(),
            "

"
        )
    }
}
