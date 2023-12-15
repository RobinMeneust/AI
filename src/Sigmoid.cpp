//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

float Sigmoid::getValue(float input) {
    return 1/(1+exp(-input));
}