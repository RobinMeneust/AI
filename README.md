## Install dependencies

- OpenCV

## Build
`cmake -S . -B build -G "Unix Makefiles"`
Then in build/ : `make`

## Run

Run the executable file in build/bin/

## Debug

- `cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug`
- `cmake --build build-debug`
- `gdb build-debug/simple_example`

## Formulae used

[PDF](AI_Project.pdf)
