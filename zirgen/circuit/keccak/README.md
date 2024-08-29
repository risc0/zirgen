# The Keccak Accelerator

This directory contains the implementation of the keccac accelerator.

## Input transcript format

The expected format of the keccak input is a list of one or more of `data blocks` ending with a `final padding` consisting of 8 bytes of zeros (`0x0000000000000000`)

### Keccak Input Transcript

| Name          | Byte Length | Description                                                                                     |
|---------------|-------------|-------------------------------------------------------------------------------------------------|
| Data Blocks   | arbitrary   | raw data blocks containing the block size followed by a sequence of bytes representing the data |
| final padding |           8 | Padding used to indicate the end of the keccak input                                            |

### Data Blocks
Data blocks represent the raw data being hashed. This contains an 2-byte value that represents the block size followed by a sequence of bytes representing the raw data, ending with a `block padding`.

| Name             | Byte Length                  | Description                                                                                                             |
|------------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| block Count (BC) |                            2 | the amount of blocks that the input spans. Each block is 136 bytes.                                                     |
| padding          |                            6 | Padding containing all 0 values. This is used to align the length along an 8-byte boundary.                             |
| raw data         | BC * 136 - 1 < N <= 136 * BC | Represents the raw data terminated by a `0x01` byte.                                                                    |
| block padding    |      BC * 136 - raw data - 1 | Padding of all 0's. Used to pad up to the current block boundary                                                        |
| block terminator |                            1 | A terminating byte for the block. This byte value represents the version. `0x80` is the only version supported for now. |

If the raw data and the `0x01` terminator fit into a block. There will be an additional block attached filled with all `0x00` ending with the `block terminator`.


### Example: using the accelerator in Rust crates

Most rust cryptographic hashing crates use an API similar to the following code snippet:
```
let mut hasher = Keccak256::new();

// write input message
hasher.update(&input);

// read hash digest
let result = hasher.finalize();
```

In this code snippet, `hasher` carries an internal state which is modified the input using the `update(input)` function. `finalize` will generate a resulting hash. In order to use the accelerator we need to generate an input transcript. As we, we need to keep track of the length of the current input data. To keep the proof under 2 million cycles the maximum amount of data we can hash in the accelerator is approximately 95 kilobytes.

The following will assume a situation where a single call to finalize is less than 95 kb of input data.

In this situation, the 