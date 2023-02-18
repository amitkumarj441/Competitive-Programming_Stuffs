// Passed all of the 6 test cases!

pub fn expected_minutes_in_oven() -> i32 {
    return 40;
}

pub fn remaining_minutes_in_oven(actual_minutes_in_oven: i32) -> i32 {
    let left_min = expected_minutes_in_oven() - actual_minutes_in_oven;
    return left_min;
}

pub fn preparation_time_in_minutes(number_of_layers: i32) -> i32 {
    fn mul(x: i32, y: i32) -> i32 {
        let x = 2;
        x*y
    }
    let spent_min = mul(2, number_of_layers);
    return spent_min;
}

pub fn elapsed_time_in_minutes(number_of_layers: i32, actual_minutes_in_oven: i32) -> i32 {
    fn add(x: i32, y: i32) -> i32 {
        x+y
    }
    return add(preparation_time_in_minutes(number_of_layers), actual_minutes_in_oven)
}
