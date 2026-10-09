import random

# Jugadores: Blancas = 1, Negras = -1
BLANCAS, NEGRAS = 1, -1

# Valores de evaluación
VALOR_PEON, VALOR_DAMA = 1.0, 2.5

# Movimientos sin captura ni coronación para declarar tablas
LIMITE_SIN_PROGRESO = 40

# Profundidad de búsqueda de alfa-beta
PROFUNDIDAD = 4

# o/O = blancas, x/X = negras (mayúscula = dama)
SIMBOLOS = {0: "·", 1: "o", 2: "O", -1: "x", -2: "X"}


# ============================================================
# REGLAS DEL JUEGO
# ============================================================

def tablero_inicial():
    """Crea el tablero 8x8 inicial (0 = vacío, ±1 = peón, ±2 = dama)."""
    T = [[0] * 8 for _ in range(8)]

    for f in range(8):
        for c in range(8):
            # Solo casillas oscuras
            if (f + c) % 2 == 1:
                if f < 3:
                    T[f][c] = -1
                elif f > 4:
                    T[f][c] = 1

    return T


def color(p):
    """Devuelve el color de una pieza: 1, -1 o 0 si está vacía."""
    return 0 if p == 0 else (1 if p > 0 else -1)


def direcciones(p):
    """Direcciones de movimiento: damas en 4 diagonales, peones solo hacia adelante."""
    if abs(p) == 2:
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if p > 0:
        return [(-1, -1), (-1, 1)]
    return [(1, -1), (1, 1)]


def dentro(f, c):
    """Comprueba si la posición está dentro del tablero."""
    return 0 <= f < 8 and 0 <= c < 8


def corona(p, f):
    """Comprueba si un peón llegó a la fila de coronación."""
    return (p == 1 and f == 0) or (p == -1 and f == 7)


def _seguir(T, f, c, p, camino, caps, salida):
    """Busca capturas encadenadas de forma recursiva."""
    hubo = False

    for df, dc in direcciones(p):
        mf, mc = f + df, c + dc          # pieza a capturar
        tf, tc = f + 2 * df, c + 2 * dc  # casilla de aterrizaje

        if dentro(tf, tc) and color(T[mf][mc]) == -color(p) and T[tf][tc] == 0:
            hubo = True

            T2 = [fila[:] for fila in T]
            T2[mf][mc] = 0

            nuevo_camino = camino + [(tf, tc)]
            nuevas_caps = caps + [(mf, mc)]

            # La coronación termina el turno
            if corona(p, tf):
                salida.append((nuevo_camino, nuevas_caps))
            else:
                _seguir(T2, tf, tc, p, nuevo_camino, nuevas_caps, salida)

    # Sin más capturas: se guarda la secuencia
    if not hubo and len(camino) > 1:
        salida.append((camino, caps))


def movimientos(estado):
    """Genera los movimientos legales (la captura es obligatoria)."""
    T, turno, _ = estado

    capturas = []
    simples = []

    for f in range(8):
        for c in range(8):
            p = T[f][c]

            if color(p) != turno:
                continue

            T2 = [fila[:] for fila in T]
            T2[f][c] = 0

            _seguir(T2, f, c, p, [(f, c)], [], capturas)

            for df, dc in direcciones(p):
                nf, nc = f + df, c + dc
                if dentro(nf, nc) and T[nf][nc] == 0:
                    simples.append(([(f, c), (nf, nc)], []))

    return capturas if capturas else simples


def aplicar(estado, mov):
    """Aplica un movimiento sobre una copia y devuelve el nuevo estado."""
    T, turno, k = estado
    camino, caps = mov

    T2 = [fila[:] for fila in T]

    (f0, c0), (f1, c1) = camino[0], camino[-1]

    p = T2[f0][c0]
    T2[f0][c0] = 0

    for (f, c) in caps:
        T2[f][c] = 0

    coronado = corona(p, f1)
    if coronado:
        p = 2 * color(p)

    T2[f1][c1] = p

    # Reinicia el contador si hubo captura o coronación
    k2 = 0 if (caps or coronado) else k + 1

    return (T2, -turno, k2)


# ============================================================
# MINIMAX CON PODA ALFA-BETA
# ============================================================

# Nodos visitados (para comparar con Minimax puro)
NODOS = 0


def evaluar(T, agente):
    """Evalúa el tablero: material de la IA - material del rival."""
    total = 0.0
    for fila in T:
        for p in fila:
            if p:
                v = VALOR_DAMA if abs(p) == 2 else VALOR_PEON
                total += v if color(p) == agente else -v
    return total


def ordenar(movs):
    """Ordena los movimientos con más capturas primero para podar más."""
    return sorted(movs, key=lambda m: -len(m[1]))


def alfabeta(estado, prof, alfa, beta, agente):
    """Minimax con poda alfa-beta."""
    global NODOS
    NODOS += 1

    T, turno, k = estado
    movs = movimientos(estado)

    # Sin movimientos pierde quien tiene el turno
    if not movs:
        return (-1000 - prof) if turno == agente else (1000 + prof)

    # Tablas por falta de progreso
    if k >= LIMITE_SIN_PROGRESO:
        return 0

    if prof == 0:
        return evaluar(T, agente)

    movs = ordenar(movs)

    if turno == agente:
        # Nodo MAX
        valor = float("-inf")
        for m in movs:
            valor = max(valor, alfabeta(aplicar(estado, m), prof - 1,
                                        alfa, beta, agente))
            alfa = max(alfa, valor)
            if alfa >= beta:
                break
        return valor

    # Nodo MIN
    valor = float("inf")
    for m in movs:
        valor = min(valor, alfabeta(aplicar(estado, m), prof - 1,
                                    alfa, beta, agente))
        beta = min(beta, valor)
        if alfa >= beta:
            break
    return valor


def mejor_jugada(estado, prof):
    """Elige al azar entre las mejores jugadas según alfa-beta."""
    global NODOS
    NODOS = 0
    turno = estado[1]

    mejor_val = float("-inf")
    mejores = []

    for m in ordenar(movimientos(estado)):
        # El margen 1e-9 evita podar jugadas empatadas
        v = alfabeta(aplicar(estado, m), prof - 1,
                     mejor_val - 1e-9, float("inf"), turno)

        if v > mejor_val:
            mejor_val, mejores = v, [m]
        elif v == mejor_val:
            mejores.append(m)

    return random.choice(mejores), mejor_val


# ============================================================
# INTERFAZ DEL JUEGO
# ============================================================

def mostrar(T):
    """Muestra el tablero en consola."""
    print("\n    " + " ".join(str(c + 1) for c in range(8)))

    for f in range(8):
        fila = [
            " " if (f + c) % 2 == 0 else SIMBOLOS[T[f][c]]
            for c in range(8)
        ]
        print(f" {f + 1} | " + " ".join(fila))

    print("(o/O = blancas, x/X = negras; mayúscula = dama)")


def texto_mov(mov):
    """Convierte un movimiento a texto."""
    camino, caps = mov

    s = "->".join(f"({f + 1},{c + 1})" for f, c in camino)

    return s + (f"  [captura {len(caps)}]" if caps else "")


def main():
    """Bucle principal de la partida."""
    print("=== Damas con Minimax + poda alfa-beta ===")

    op = ""
    while op not in ("b", "n"):
        op = input("Tu color (b = blancas, n = negras): ").strip().lower()

    humano = BLANCAS if op == "b" else NEGRAS
    prof = PROFUNDIDAD

    estado = (tablero_inicial(), BLANCAS, 0)

    # Cuenta repeticiones de posición
    vistos = {}

    while True:
        T, turno, k = estado

        mostrar(T)

        nombre = "Blancas" if turno == BLANCAS else "Negras"

        clave = (tuple(map(tuple, T)), turno)
        vistos[clave] = vistos.get(clave, 0) + 1

        movs = movimientos(estado)

        if not movs:
            ganan = "Negras" if turno == BLANCAS else "Blancas"
            print(f"\n{nombre} no tienen movimientos. ¡Ganan {ganan}!")
            break

        if k >= LIMITE_SIN_PROGRESO:
            print("\nTablas: 40 jugadas sin captura ni coronación.")
            break

        if vistos[clave] >= 3:
            print("\nTablas: la posición se repitió 3 veces.")
            break

        # Turno del humano
        if turno == humano:
            print(f"\nTurno de {nombre}. Movimientos legales:")

            for i, m in enumerate(movs, 1):
                print(f"  {i}. {texto_mov(m)}")

            while True:
                try:
                    mov = movs[int(input("Elige el número: ")) - 1]
                    break
                except (ValueError, IndexError):
                    print("Opción inválida.")

        # Turno de la IA
        else:
            mov, val = mejor_jugada(estado, prof)

            print(
                f"\nIA ({nombre}) juega: {texto_mov(mov)}   "
                f"(valor = {val}, nodos visitados = {NODOS})"
            )

        estado = aplicar(estado, mov)


if __name__ == "__main__":
    main()
