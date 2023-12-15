## Install dependencies


## Build
`cmake -S . -B build -G "Unix Makefiles"`
Then in build/ : `make`

## Run


## Debug

- `cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug`
- `cmake --build build-debug`
- `gdb build-debug/simple_example`