import os
import argparse
import time
import statistics
import csv

# ==================================================
# Parámetros globales
# ==================================================

WARMUP_ITERS = 10
MEASURE_ITERS = 50

# ==================================================
# Parseo de argumentos (ANTES de importar torch)
# ==================================================

parser = argparse.ArgumentParser(
    description="Benchmark CPU/GPU: GEMM y GEMV con distintos dtypes"
)
parser.add_argument(
    "--device",
    choices=["cpu", "gpu"],
    default="cpu",
    help="Dispositivo de ejecución"
)
parser.add_argument(
    "--dtype",
    choices=["fp32", "fp16", "bf16"],
    default="fp32",
    help="Tipo de dato"
)
args = parser.parse_args()

# ==================================================
# Configuración de hilos CPU (solo relevante en CPU)
# ==================================================

CPU_THREADS = int(os.environ.get("CPU_THREADS", "1"))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

import torch

# ==================================================
# Selección de dtype
# ==================================================

if args.dtype == "fp32":
    TORCH_DTYPE = torch.float32
elif args.dtype == "fp16":
    TORCH_DTYPE = torch.float16
elif args.dtype == "bf16":
    TORCH_DTYPE = torch.bfloat16
else:
    raise RuntimeError("dtype no soportado")

# ==================================================
# Validación de combinaciones
# ==================================================

if args.device == "cpu":
    if args.dtype != "fp32":
        raise RuntimeError(
            "En CPU solo se permite fp32 para benchmarking fiable"
        )

if args.device == "gpu":
    if args.dtype == "bf16":
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError(
                "BF16 requiere GPU Ampere o posterior"
            )

# ==================================================
# Configuración TF32 (solo Ampere+)
# ==================================================

if args.device == "gpu":
    if args.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

# ==================================================
# Utilidades
# ==================================================

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_op(op, device):
    times = []

    # Warm-up
    for _ in range(WARMUP_ITERS):
        op()
    sync(device)

    # Medición
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        op()
        sync(device)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return mean_time, std_time


# ==================================================
# Benchmarks
# ==================================================

def benchmark_matmul(N, device):
    A = torch.randn((N, N), device=device, dtype=TORCH_DTYPE)
    B = torch.randn((N, N), device=device, dtype=TORCH_DTYPE)

    def op():
        return A @ B

    mean_t, std_t = time_op(op, device)
    flops = 2 * N**3
    gflops = flops / mean_t / 1e9
    return mean_t, std_t, gflops


def benchmark_matvec(N, device):
    A = torch.randn((N, N), device=device, dtype=TORCH_DTYPE)
    x = torch.randn((N,), device=device, dtype=TORCH_DTYPE)

    def op():
        return A @ x

    mean_t, std_t = time_op(op, device)
    flops = 2 * N**2
    gflops = flops / mean_t / 1e9
    return mean_t, std_t, gflops


# ==================================================
# Programa principal
# ==================================================

def main():
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    results = []

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible en este sistema")
        device = torch.device("cuda")
        label = f"gpu_cuda_{args.dtype}"
        label_pretty = "GPU (CUDA)"
    else:
        device = torch.device("cpu")
        label = f"cpu_{CPU_THREADS}T_{args.dtype}"
        label_pretty = (
            f"CPU ({CPU_THREADS} thread/s)"
        )
        
    output_csv = f"benchmark_{label}.csv"
 
    print("==============================================")
    print(" Benchmark de GEMM-GEMV (CPU/GPU)")
    print("==============================================")
    print(f"Dispositivo: {label_pretty}")
    print(f"Tipo de dato: {args.dtype}")
    print(f"Warm-up: {WARMUP_ITERS} iteraciones")
    print(f"Mediciones: {MEASURE_ITERS} iteraciones")
    print(f"Salida CSV: {output_csv}")
    
    for N in sizes:
        print(f"\n--- Tamaño N = {N} ---")

        t_mean, t_std, gflops = benchmark_matmul(N, device)
        print(
            f"MATMUL (GEMM) | "
            f"Tiempo medio = {t_mean:.6f} s "
            f"(± {t_std:.6f}) | "
            f"{gflops:.2f} GFLOP/s"
        )
        results.append(["matmul", N, label, t_mean, t_std, gflops])

        t_mean, t_std, gflops = benchmark_matvec(N, device)
        print(
            f"MATVEC (GEMV) | "
            f"Tiempo medio = {t_mean:.6f} s "
            f"(± {t_std:.6f}) | "
            f"{gflops:.2f} GFLOP/s"
        )
        results.append(["matvec", N, label, t_mean, t_std, gflops])

    # -------------------------
    # Guardar CSV
    # -------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ["operation", "N", "config", "time_mean_s", "time_std_s", "gflops"]
        ])
        writer.writerows(results)

    print(f"\nResultados guardados en: {output_csv}")


if __name__ == "__main__":
    main()
