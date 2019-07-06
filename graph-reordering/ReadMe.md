## Compile

* on KNL

```
mkdir -p build & cd build
cmake .. -DKNL=ON
make -j
```

* on CPU

```zsh
mkdir -p build & cd build
cmake .. 
make -j
```

## Usage

### Porder (We used in our experiments)

```zsh
./reorder /mnt/storage1/yche/datasets/snap_livejournal -order gro
```

* orders: hybrid, mloggapa, slashburn, bfsr, dfs, gro