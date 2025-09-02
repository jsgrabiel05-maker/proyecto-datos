# app.py
# Streamlit Pokédex Avanzada
# Requisitos: streamlit, pandas

import streamlit as st
import pandas as pd
import numpy as np

POSSIBLE_COLS = {
    "name": ["Name", "Pokemon", "nombre", "Nombre"],
    "type1": ["Type 1", "Primary Type", "Tipo 1", "Tipo1", "type1"],
    "type2": ["Type 2", "Secondary Type", "Tipo 2", "Tipo2", "type2"],
    "hp": ["HP", "Hp", "hp"],
    "attack": ["Attack", "Atk", "Ataque", "attack"],
    "defense": ["Defense", "Defensa", "Def", "defense"],
    "sp_atk": ["Sp. Atk", "SpAtk", "Special Attack", "Ataque Especial", "sp_atk", "SpAtk"],
    "sp_def": ["Sp. Def", "SpDef", "Special Defense", "Defensa Especial", "sp_def", "SpDef"],
    "speed": ["Speed", "Velocidad", "speed"],
    "stage": ["Stage", "EvolutionStage", "Etapa", "EvolStage", "Evolucion", "evolution_stage"],
    "generation": ["Generation", "Generación", "Gen", "generation"],
    "legendary": ["Legendary", "Legendario", "is_legendary"]
}

def find_col(df: pd.DataFrame, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

@st.cache_data
def load_data(path: str = "pokemon.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalización mínima: quitar espacios en nombres de columnas
    df.columns = [c.strip() for c in df.columns]
    return df

# ==============================
# Tabla de efectividades (ataque -> defensa)
# valores: 2.0 super efectivo, 0.5 poco efectivo, 0.0 inmune, 1.0 normal
# Basado en las reglas clásicas (Generaciones modernas, sin Megas/formas regionales).
# ==============================

TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground",
    "Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"
]

# Mapa base: para cada atacante, contra cada defensor
# (fuente clásica consolidada; si cambian reglas en gens futuras, ajustar aquí)
TYPE_CHART = {
    "Normal":   {"Rock":0.5,"Ghost":0.0,"Steel":0.5},
    "Fire":     {"Fire":0.5,"Water":0.5,"Grass":2.0,"Ice":2.0,"Bug":2.0,"Rock":0.5,"Dragon":0.5,"Steel":2.0},
    "Water":    {"Fire":2.0,"Water":0.5,"Grass":0.5,"Ground":2.0,"Rock":2.0,"Dragon":0.5},
    "Electric": {"Water":2.0,"Electric":0.5,"Grass":0.5,"Ground":0.0,"Flying":2.0,"Dragon":0.5},
    "Grass":    {"Fire":0.5,"Water":2.0,"Grass":0.5,"Poison":0.5,"Ground":2.0,"Flying":0.5,"Bug":0.5,"Rock":2.0,"Dragon":0.5,"Steel":0.5},
    "Ice":      {"Fire":0.5,"Water":0.5,"Grass":2.0,"Ground":2.0,"Flying":2.0,"Dragon":2.0,"Steel":0.5},
    "Fighting": {"Normal":2.0,"Ice":2.0,"Poison":0.5,"Flying":0.5,"Psychic":0.5,"Bug":0.5,"Rock":2.0,"Ghost":0.0,"Dark":2.0,"Steel":2.0,"Fairy":0.5},
    "Poison":   {"Grass":2.0,"Poison":0.5,"Ground":0.5,"Rock":0.5,"Ghost":0.5,"Steel":0.0,"Fairy":2.0},
    "Ground":   {"Fire":2.0,"Electric":2.0,"Grass":0.5,"Poison":2.0,"Flying":0.0,"Bug":0.5,"Rock":2.0,"Steel":2.0},
    "Flying":   {"Electric":0.5,"Grass":2.0,"Fighting":2.0,"Bug":2.0,"Rock":0.5,"Steel":0.5},
    "Psychic":  {"Fighting":2.0,"Poison":2.0,"Psychic":0.5,"Dark":0.0,"Steel":0.5},
    "Bug":      {"Fire":0.5,"Grass":2.0,"Fighting":0.5,"Poison":0.5,"Flying":0.5,"Psychic":2.0,"Ghost":0.5,"Dark":2.0,"Steel":0.5,"Fairy":0.5},
    "Rock":     {"Fire":2.0,"Ice":2.0,"Fighting":0.5,"Ground":0.5,"Flying":2.0,"Bug":2.0,"Steel":0.5},
    "Ghost":    {"Normal":0.0,"Psychic":2.0,"Ghost":2.0,"Dark":0.5},
    "Dragon":   {"Dragon":2.0,"Steel":0.5,"Fairy":0.0},
    "Dark":     {"Fighting":0.5,"Psychic":2.0,"Ghost":2.0,"Dark":0.5,"Fairy":0.5},
    "Steel":    {"Fire":0.5,"Water":0.5,"Electric":0.5,"Ice":2.0,"Rock":2.0,"Fairy":2.0,"Steel":0.5},
    "Fairy":    {"Fire":0.5,"Fighting":2.0,"Poison":0.5,"Dragon":2.0,"Dark":2.0,"Steel":0.5}
}

def effectiveness(attacking: str, defending: str) -> float:
    """Multiplicador de ataque 'attacking' contra tipo 'defending'."""
    if attacking not in TYPE_CHART:
        return 1.0
    return TYPE_CHART[attacking].get(defending, 1.0)

def combined_defense_multiplier(attacking: str, defend_types: list[str]) -> float:
    """Multiplicador resultante al atacar a un Pokémon con uno o dos tipos."""
    mult = 1.0
    for t in defend_types:
        if t and isinstance(t, str) and t.strip():
            mult *= effectiveness(attacking, t)
    return mult

def defensive_profile(type1: str, type2: str | None):
    """Regresa dict con inmunidades (0x), resistencias (<1x), neutrales (=1x) y debilidades (>1x)."""
    tlist = [type1] + ([type2] if type2 and pd.notna(type2) and str(type2).strip() else [])
    profile = {"0x":[],"0.25x":[],"0.5x":[],"1x":[],"2x":[],"4x":[]}
    for atk in TYPES:
        m = combined_defense_multiplier(atk, tlist)
        if np.isclose(m, 0.0):
            profile["0x"].append(atk)
        elif np.isclose(m, 0.25):
            profile["0.25x"].append(atk)
        elif np.isclose(m, 0.5):
            profile["0.5x"].append(atk)
        elif np.isclose(m, 1.0):
            profile["1x"].append(atk)
        elif np.isclose(m, 2.0):
            profile["2x"].append(atk)
        elif np.isclose(m, 4.0):
            profile["4x"].append(atk)
        else:
            if m < 0.5: profile["0.25x"].append(atk)
            elif m < 1.0: profile["0.5x"].append(atk)
            elif m < 2.0: profile["1x"].append(atk)
            elif m < 4.0: profile["2x"].append(atk)
            else: profile["4x"].append(atk)
    return profile


st.set_page_config(page_title="Pokédex Avanzada", page_icon=" ", layout="wide")
st.title(" Pokédex Avanzada (Streamlit)")

default_file_loaded = False
df = None

uploaded = st.sidebar.file_uploader("Sube tu archivo CSV de Pokémon", type=["csv"])
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_data("pokemon.csv")
        default_file_loaded = True
except Exception as e:
    st.error(f"No se pudo cargar el CSV. Detalle: {e}")
    st.stop()

# Descubrir columnas relevantes de manera robusta
COL_NAME = find_col(df, POSSIBLE_COLS["name"])
COL_T1 = find_col(df, POSSIBLE_COLS["type1"])
COL_T2 = find_col(df, POSSIBLE_COLS["type2"])
COL_HP = find_col(df, POSSIBLE_COLS["hp"])
COL_ATK = find_col(df, POSSIBLE_COLS["attack"])
COL_DEF = find_col(df, POSSIBLE_COLS["defense"])
COL_SPA = find_col(df, POSSIBLE_COLS["sp_atk"])
COL_SPD = find_col(df, POSSIBLE_COLS["sp_def"])
COL_SPE = find_col(df, POSSIBLE_COLS["speed"])
COL_STAGE = find_col(df, POSSIBLE_COLS["stage"])

required_basic = [COL_NAME, COL_T1, COL_HP, COL_ATK, COL_DEF, COL_SPA, COL_SPD, COL_SPE]
if any(c is None for c in required_basic):
    st.warning("⚠️ No se encontraron todas las columnas esperadas. "
               "Asegúrate de que tu CSV tenga al menos: "
               "`Name`, `Type 1`, `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed` "
               "(o nombres equivalentes).")
    st.write("Columnas detectadas:", list(df.columns))

# Normalizamos algunos tipos de datos por si vienen como string
for c in [COL_HP, COL_ATK, COL_DEF, COL_SPA, COL_SPD, COL_SPE]:
    if c and df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Tabs principales
tab_search, tab_types, tab_compare, tab_advanced = st.tabs(
    ["🔎 Buscador básico", "🧪 Análisis de tipos", "⚖️ Comparar Pokémon", "🧰 Búsqueda avanzada"]
)

# ==============================
# 🔎 Buscador básico
# ==============================
with tab_search:
    st.subheader("Buscar por nombre y tipo")
    colA, colB, colC = st.columns([2,2,1])

    with colA:
        q = st.text_input("Nombre contiene...", value="")
    with colB:
        # Tipos disponibles en el CSV (columna Type 1 y Type 2)
        tipos_csv = set()
        if COL_T1: tipos_csv.update(df[COL_T1].dropna().unique().tolist())
        if COL_T2: tipos_csv.update(df[COL_T2].dropna().unique().tolist())
        tipos_csv = sorted([t for t in tipos_csv if isinstance(t, str)])
        tipos_sel = st.multiselect("Filtrar por tipo (uno o ambos)", tipos_csv)
    with colC:
        mostrar_cols = st.multiselect(
            "Columnas a mostrar",
            [COL_NAME, COL_T1, COL_T2, COL_HP, COL_ATK, COL_DEF, COL_SPA, COL_SPD, COL_SPE],
            default=[COL_NAME, COL_T1, COL_T2, COL_ATK, COL_SPE]
        )

    res = df.copy()
    if q and COL_NAME:
        res = res[res[COL_NAME].str.contains(q, case=False, na=False)]

    if tipos_sel and (COL_T1 or COL_T2):
        mask = False
        if COL_T1:
            mask = res[COL_T1].isin(tipos_sel)
        if COL_T2:
            mask = mask | res[COL_T2].isin(tipos_sel) if isinstance(mask, pd.Series) else res[COL_T2].isin(tipos_sel)
        res = res[mask]

    st.write(f"Resultados: {len(res)}")
    if mostrar_cols:
        st.dataframe(res[mostrar_cols])
    else:
        st.dataframe(res)

#  Análisis de tipos
with tab_types:
    st.subheader("Fortalezas, resistencias e inmunidades por tipo")
    if COL_NAME is None or COL_T1 is None:
        st.info("No se detectaron columnas de Nombre y/o Tipo 1.")
    else:
        poke_opt = st.selectbox("Selecciona un Pokémon", df[COL_NAME].dropna().sort_values().unique())
        sel = df[df[COL_NAME] == poke_opt].iloc[0]
        t1 = str(sel[COL_T1]) if pd.notna(sel[COL_T1]) else None
        t2 = str(sel[COL_T2]) if (COL_T2 and pd.notna(sel[COL_T2])) else None

        st.write(f"**Tipos:** {t1}" + (f" / {t2}" if t2 else ""))

        prof = defensive_profile(t1, t2)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Debilidades** (2× / 4×)")
            st.write("2×:", ", ".join(prof["2x"]) if prof["2x"] else "—")
            st.write("4×:", ", ".join(prof["4x"]) if prof["4x"] else "—")
        with c2:
            st.markdown("**Resistencias** (½× / ¼×)")
            st.write("½×:", ", ".join(prof["0.5x"]) if prof["0.5x"] else "—")
            st.write("¼×:", ", ".join(prof["0.25x"]) if prof["0.25x"] else "—")
        with c3:
            st.markdown("**Inmunidades** (0×)")
            st.write(", ".join(prof["0x"]) if prof["0x"] else "—")

        # Tabla completa de multiplicadores defensivos contra cada tipo atacante
        mults = {atk: combined_defense_multiplier(atk, [t1] + ([t2] if t2 else [])) for atk in TYPES}
        df_mults = pd.DataFrame({"Ataque": list(mults.keys()), "Multiplicador": list(mults.values())})
        st.markdown("**Detalle completo de multiplicadores (ataque → este Pokémon)**")
        st.dataframe(df_mults.sort_values("Multiplicador", ascending=False), use_container_width=True)


#Comparar Pokémon
with tab_compare:
    st.subheader("Comparativa de estadísticas")
    if any(c is None for c in [COL_NAME, COL_HP, COL_ATK, COL_DEF, COL_SPA, COL_SPD, COL_SPE]):
        st.info("No se detectaron todas las columnas de estadísticas necesarias.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.selectbox("Pokémon A", df[COL_NAME].dropna().sort_values().unique(), key="cmpA")
        with c2:
            p2 = st.selectbox("Pokémon B", df[COL_NAME].dropna().sort_values().unique(), key="cmpB")

        if p1 == p2:
            st.warning("Selecciona dos Pokémon distintos.")
        else:
            s1 = df[df[COL_NAME] == p1].iloc[0]
            s2 = df[df[COL_NAME] == p2].iloc[0]

            stats = [
                ("HP", COL_HP),
                ("Attack", COL_ATK),
                ("Defense", COL_DEF),
                ("Sp. Atk", COL_SPA),
                ("Sp. Def", COL_SPD),
                ("Speed", COL_SPE),
            ]

            rows = []
            for label, col in stats:
                v1 = float(s1[col]) if pd.notna(s1[col]) else np.nan
                v2 = float(s2[col]) if pd.notna(s2[col]) else np.nan
                if np.isnan(v1) or np.isnan(v2):
                    winner = "—"
                    diff = "—"
                else:
                    if v1 > v2:
                        winner = f"{p1}"
                    elif v2 > v1:
                        winner = f"{p2}"
                    else:
                        winner = "Empate"
                    diff = int(abs(v1 - v2))
                rows.append([label, v1, v2, winner, diff])

            cmp_df = pd.DataFrame(rows, columns=["Estadística", p1, p2, "Mayor", "Diferencia"])
            st.dataframe(cmp_df, use_container_width=True)

            # Resumen textual
            bullets = []
            for _, label, v1, v2, winner, _diff in cmp_df.itertuples():
                if winner == "Empate":
                    bullets.append(f"• {label}: empate ({int(v1)} = {int(v2)})")
                elif winner == p1:
                    bullets.append(f"• {label}: {p1} ({int(v1)} > {int(v2)})")
                elif winner == p2:
                    bullets.append(f"• {label}: {p2} ({int(v2)} > {int(v1)})")
            st.markdown("**Resumen:**")
            st.write("\n".join(bullets))

#Búsqueda avanzada
with tab_advanced:
    st.subheader("Filtros por características específicas")

    # Controles para nombre y tipos
    colX, colY, colZ = st.columns([2,2,2])
    with colX:
        q2 = st.text_input("Nombre contiene...", value="", key="adv_name")
    with colY:
        tipos_sel2 = st.multiselect("Tipos", tipos_csv, key="adv_types")
    with colZ:
        stage_options = []
        if COL_STAGE:
            stage_options = ["(todos)"] + sorted([str(x) for x in df[COL_STAGE].dropna().unique()])
            stage_val = st.selectbox("Etapa evolutiva", options=stage_options)
        else:
            stage_val = "(no disponible en CSV)"
            st.caption("No se encontró columna de etapa evolutiva (Stage/EvolutionStage).")

    # control de estadisticas
    def stat_filter_ui(label, colname, key_prefix):
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            op = st.selectbox(label, ["(ignorar)", "≥", "≤", "="], key=f"{key_prefix}_op")
        with c2:
            val = st.number_input("valor", min_value=0, max_value=300, value=50, step=1, key=f"{key_prefix}_val")
        with c3:
            st.caption(f"Columna: {colname if colname else '—'}")
        return op, val

    st.markdown("**Filtros de estadísticas**")
    op_hp, val_hp = stat_filter_ui("HP", COL_HP, "hp")
    op_atk, val_atk = stat_filter_ui("Attack", COL_ATK, "atk")
    op_def, val_def = stat_filter_ui("Defense", COL_DEF, "def")
    op_spa, val_spa = stat_filter_ui("Sp. Atk", COL_SPA, "spa")
    op_spd, val_spd = stat_filter_ui("Sp. Def", COL_SPD, "spd")
    op_spe, val_spe = stat_filter_ui("Speed", COL_SPE, "spe")

    res2 = df.copy()

    # Nombre
    if q2 and COL_NAME:
        res2 = res2[res2[COL_NAME].str.contains(q2, case=False, na=False)]

    # Tipos
    if tipos_sel2 and (COL_T1 or COL_T2):
        mask2 = False
        if COL_T1:
            mask2 = res2[COL_T1].isin(tipos_sel2)
        if COL_T2:
            mask2 = mask2 | res2[COL_T2].isin(tipos_sel2) if isinstance(mask2, pd.Series) else res2[COL_T2].isin(tipos_sel2)
        res2 = res2[mask2]

    # Etapa evolutiva (si existe)
    if COL_STAGE and stage_options and stage_val != "(todos)":
        res2 = res2[res2[COL_STAGE].astype(str) == stage_val]

    # Función para aplicar operador a una columna
    def apply_op(data, col, op, val):
        if col is None or op == "(ignorar)":
            return data
        if op == "≥":
            return data[data[col] >= val]
        if op == "≤":
            return data[data[col] <= val]
        if op == "=":
            return data[data[col] == val]
        return data

    # Aplicamos filtros por stats
    res2 = apply_op(res2, COL_HP, op_hp, val_hp)
    res2 = apply_op(res2, COL_ATK, op_atk, val_atk)
    res2 = apply_op(res2, COL_DEF, op_def, val_def)
    res2 = apply_op(res2, COL_SPA, op_spa, val_spa)
    res2 = apply_op(res2, COL_SPD, op_spd, val_spd)
    res2 = apply_op(res2, COL_SPE, op_spe, val_spe)

    # Mostrar
    default_cols = [c for c in [COL_NAME, COL_T1, COL_T2, COL_HP, COL_ATK, COL_DEF, COL_SPA, COL_SPD, COL_SPE, COL_STAGE] if c]
    st.write(f"Resultados: {len(res2)}")
    st.dataframe(res2[default_cols] if default_cols else res2, use_container_width=True)

# Footer
st.caption("Consejo: si tus nombres de columnas difieren, ajusta la lista POSSIBLE_COLS al inicio del archivo.")
