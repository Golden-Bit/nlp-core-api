#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/build_app}"
PERSIST_ROOT="${PERSIST_ROOT:-/persist}"

# Le due cartelle che vuoi rendere persistenti
DIRS=("data_stores" "vector_stores")

mkdir -p "$PERSIST_ROOT"

seed_dir() {
  local src="$1"
  local dst="$2"

  mkdir -p "$dst"

  # Modalità reseed controllata: se RESEED=1, resetta il contenuto del volume e ricopia dal codice
  if [ "${RESEED:-0}" = "1" ]; then
    rm -rf "${dst:?}/"*
    if [ -d "$src" ]; then
      cp -a "$src/." "$dst/" || true
    fi
    return
  fi

  # Modalità sicura (default): copia SOLO i file mancanti (non sovrascrive quelli già presenti nel volume)
  # Utile se, rebuildando, vuoi aggiungere nuovi file “di default” senza perdere lo stato runtime.
  if [ -d "$src" ]; then
    cp -an "$src/." "$dst/" || true
  fi
}

for d in "${DIRS[@]}"; do
  src_in_app="${APP_DIR}/${d}"
  dst_in_persist="${PERSIST_ROOT}/${d}"

  # 1) Seed (merge non distruttivo) dal contenuto dell'immagine -> volume persistente
  seed_dir "$src_in_app" "$dst_in_persist"

  # 2) Symlink assoluti: /data_stores e /vector_stores -> /persist/...
  ln -sfn "$dst_in_persist" "/${d}"

  # 3) Symlink anche dentro la root dell'app (per path relativi che diventano /build_app/data_stores)
  #    Se esiste una directory reale in immagine, la rimuoviamo (dopo il seed) e la rimpiazziamo col link.
  if [ -e "$src_in_app" ] && [ ! -L "$src_in_app" ]; then
    rm -rf "$src_in_app"
  fi
  ln -sfn "/${d}" "$src_in_app"
done

exec "$@"
