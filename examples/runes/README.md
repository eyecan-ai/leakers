# Generate Runes



```
leakers runes generate -c ${CONFIG} --image_size 64 -n ${NAME} --device gpu -b ${BIT_SIZE}
```

for example , to generate default runes, run:

```
leakers runes generate -c runes_cfg.yml --image_size 64 -n runes --device gpu -b 6
```

The output will be a `XXX.rune` file suitable for inference.



# Debug

## Printe a Runes board


```
leakers runes debug print_board -m ${RUNE_FILE} --cpu -o ${OUTPUT_IMAGE}
```


## Load Runes in a board for interactive debug


Debug on a virtual image filled with a standard board

```
leakers runes debug image -m runes.rune -w 12 -h 8
```

Or use a target image to debug on

```
leakers runes debug image -m runes.rune -i ${TARGET_IMAGE}
```

Try to generate the `${TARGET_IMAGE}` with printed board and Blender
