// TODO: Unlike in the naive version, special-case the center of the stencil to
//       avoid all the convoluted indexing. Extract the kernel of the naive
//       version and defer to it on the edge.

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
