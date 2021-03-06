use super::*;

pub extern "C" fn io_writeln(
    _: &Runtime,
    _: ObjectPointer,
    args: &[ObjectPointer],
) -> Result<ObjectPointer, ObjectPointer> {
    for (i, arg) in args.iter().enumerate() {
        print!("{}", arg.to_string());
        if i != args.len() - 1 {
            print!(" ");
        }
    }
    println!();
    Ok(ObjectPointer::number(args.len() as f64))
}

pub extern "C" fn io_write(
    _: &Runtime,
    _: ObjectPointer,
    args: &[ObjectPointer],
) -> Result<ObjectPointer, ObjectPointer> {
    for (i, arg) in args.iter().enumerate() {
        print!("{}", arg.to_string());
        if i != args.len() - 1 {
            print!(" ");
        }
    }
    Ok(ObjectPointer::number(args.len() as f64))
}

pub(super) fn register_io(state: &mut RcState) {
    let io_object = state.gc.allocate(Object::new(ObjectValue::None));
    io_object.add_attribute(
        &Arc::new(String::from("writeln")),
        new_native_fn(state, io_writeln, -1),
    );
    io_object.add_attribute(
        &Arc::new(String::from("write")),
        new_native_fn(state, io_write, -1),
    );

    state.static_variables.insert("io".to_owned(), io_object);
}
