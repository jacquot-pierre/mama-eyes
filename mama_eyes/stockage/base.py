"""Base SQLite : metadonnees, parametres et mesures.

La base ne stocke jamais de pixels. Les images restent des fichiers ; SQLite
porte les dates, la qualite de recalage, les reperes et les mesures.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from ..domain import (
    Cliche,
    Druse,
    Evenement,
    QualiteRecalage,
    Reperes,
    Seance,
)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS seance (
  id      INTEGER PRIMARY KEY,
  oeil    TEXT NOT NULL CHECK (oeil IN ('D','G')),
  date    TEXT NOT NULL,
  note    TEXT,
  UNIQUE (oeil, date)
);

CREATE TABLE IF NOT EXISTS cliche (
  id          INTEGER PRIMARY KEY,
  seance_id   INTEGER NOT NULL REFERENCES seance(id) ON DELETE CASCADE,
  fichier     TEXT NOT NULL,
  sha256      TEXT NOT NULL UNIQUE,
  largeur     INTEGER NOT NULL,
  hauteur     INTEGER NOT NULL,
  nettete     REAL,
  exclu       INTEGER NOT NULL DEFAULT 0,
  importe_le  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cliche_seance ON cliche(seance_id);

CREATE TABLE IF NOT EXISTS evenement (
  id      INTEGER PRIMARY KEY,
  date    TEXT NOT NULL,
  libelle TEXT NOT NULL,
  note    TEXT
);

CREATE TABLE IF NOT EXISTS reperes (
  oeil            TEXT PRIMARY KEY CHECK (oeil IN ('D','G')),
  disque_x        REAL NOT NULL,
  disque_y        REAL NOT NULL,
  disque_rayon_px REAL NOT NULL,
  macula_x        REAL NOT NULL,
  macula_y        REAL NOT NULL,
  manuel          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS recalage (
  cliche_id   INTEGER PRIMARY KEY REFERENCES cliche(id) ON DELETE CASCADE,
  empreinte   TEXT NOT NULL,
  matrice     TEXT NOT NULL,
  inliers     INTEGER NOT NULL,
  residu_rms  REAL NOT NULL,
  valide      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS analyse (
  id          INTEGER PRIMARY KEY,
  oeil        TEXT NOT NULL,
  methode     TEXT NOT NULL,
  params_json TEXT NOT NULL,
  empreinte   TEXT NOT NULL,
  cree_le     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mesure (
  analyse_id  INTEGER NOT NULL REFERENCES analyse(id) ON DELETE CASCADE,
  seance_id   INTEGER NOT NULL REFERENCES seance(id) ON DELETE CASCADE,
  nombre      INTEGER NOT NULL,
  surface_px  INTEGER NOT NULL,
  surface_mm2 REAL,
  apparues    INTEGER NOT NULL DEFAULT 0,
  disparues   INTEGER NOT NULL DEFAULT 0,
  fusions     INTEGER NOT NULL DEFAULT 0,
  scissions   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (analyse_id, seance_id)
);

CREATE TABLE IF NOT EXISTS druse (
  id            INTEGER PRIMARY KEY,
  analyse_id    INTEGER NOT NULL REFERENCES analyse(id) ON DELETE CASCADE,
  seance_id     INTEGER NOT NULL REFERENCES seance(id) ON DELETE CASCADE,
  piste_id      INTEGER,
  cx            REAL NOT NULL,
  cy            REAL NOT NULL,
  surface_px    INTEGER NOT NULL,
  excentricite  REAL,
  solidite      REAL,
  intensite_moy REAL
);
CREATE INDEX IF NOT EXISTS idx_druse_analyse ON druse(analyse_id, seance_id);
CREATE INDEX IF NOT EXISTS idx_druse_piste ON druse(analyse_id, piste_id);
"""


def _en_date(valeur: str) -> date:
    return datetime.strptime(valeur, "%Y-%m-%d").date()


class Base:
    """Acces SQLite. Utilisable comme gestionnaire de contexte."""

    def __init__(self, chemin: Path | str):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.cx = sqlite3.connect(self.chemin, check_same_thread=False)
        self.cx.row_factory = sqlite3.Row
        self.cx.executescript(SCHEMA)
        self.cx.commit()

    def __enter__(self) -> "Base":
        return self

    def __exit__(self, *exc) -> None:
        self.fermer()

    def fermer(self) -> None:
        self.cx.close()

    # -- seances -------------------------------------------------------------

    def seance_id(self, oeil: str, jour: date, creer: bool = True) -> int | None:
        cur = self.cx.execute(
            "SELECT id FROM seance WHERE oeil = ? AND date = ?", (oeil, jour.isoformat())
        )
        ligne = cur.fetchone()
        if ligne:
            return ligne["id"]
        if not creer:
            return None
        cur = self.cx.execute(
            "INSERT INTO seance (oeil, date) VALUES (?, ?)", (oeil, jour.isoformat())
        )
        self.cx.commit()
        return cur.lastrowid

    def seances(self, oeil: str, avec_cliches: bool = True) -> list[Seance]:
        lignes = self.cx.execute(
            "SELECT * FROM seance WHERE oeil = ? ORDER BY date", (oeil,)
        ).fetchall()
        resultat = [
            Seance(id=l["id"], oeil=l["oeil"], date=_en_date(l["date"]), note=l["note"])
            for l in lignes
        ]
        if avec_cliches and resultat:
            par_id = {s.id: s for s in resultat}
            for c in self.cx.execute(
                "SELECT c.* FROM cliche c JOIN seance s ON s.id = c.seance_id "
                "WHERE s.oeil = ? ORDER BY c.fichier",
                (oeil,),
            ):
                seance = par_id.get(c["seance_id"])
                if seance is not None:
                    seance.cliches.append(
                        Cliche(
                            id=c["id"],
                            seance_id=c["seance_id"],
                            fichier=c["fichier"],
                            sha256=c["sha256"],
                            largeur=c["largeur"],
                            hauteur=c["hauteur"],
                            nettete=c["nettete"],
                            exclu=bool(c["exclu"]),
                            importe_le=c["importe_le"],
                        )
                    )
        return resultat

    def definir_note_seance(self, seance_id: int, note: str | None) -> None:
        self.cx.execute("UPDATE seance SET note = ? WHERE id = ?", (note, seance_id))
        self.cx.commit()

    # -- cliches -------------------------------------------------------------

    def sha_connu(self, sha256: str) -> bool:
        return (
            self.cx.execute(
                "SELECT 1 FROM cliche WHERE sha256 = ?", (sha256,)
            ).fetchone()
            is not None
        )

    def ajouter_cliche(self, cliche: Cliche) -> int:
        cur = self.cx.execute(
            "INSERT INTO cliche (seance_id, fichier, sha256, largeur, hauteur, "
            "nettete, exclu, importe_le) VALUES (?,?,?,?,?,?,?,?)",
            (
                cliche.seance_id,
                cliche.fichier,
                cliche.sha256,
                cliche.largeur,
                cliche.hauteur,
                cliche.nettete,
                int(cliche.exclu),
                cliche.importe_le or datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.cx.commit()
        return cur.lastrowid

    def definir_exclusion(self, cliche_id: int, exclu: bool) -> None:
        self.cx.execute(
            "UPDATE cliche SET exclu = ? WHERE id = ?", (int(exclu), cliche_id)
        )
        self.cx.commit()

    def supprimer_cliche(self, cliche_id: int) -> None:
        self.cx.execute("DELETE FROM cliche WHERE id = ?", (cliche_id,))
        self.cx.commit()

    # -- evenements ----------------------------------------------------------

    def ajouter_evenement(self, jour: date, libelle: str, note: str | None = None) -> int:
        cur = self.cx.execute(
            "INSERT INTO evenement (date, libelle, note) VALUES (?,?,?)",
            (jour.isoformat(), libelle, note),
        )
        self.cx.commit()
        return cur.lastrowid

    def evenements(self) -> list[Evenement]:
        return [
            Evenement(
                id=l["id"],
                date=_en_date(l["date"]),
                libelle=l["libelle"],
                note=l["note"],
            )
            for l in self.cx.execute("SELECT * FROM evenement ORDER BY date")
        ]

    def supprimer_evenement(self, evenement_id: int) -> None:
        self.cx.execute("DELETE FROM evenement WHERE id = ?", (evenement_id,))
        self.cx.commit()

    # -- reperes -------------------------------------------------------------

    def enregistrer_reperes(self, r: Reperes) -> None:
        self.cx.execute(
            "INSERT INTO reperes (oeil, disque_x, disque_y, disque_rayon_px, "
            "macula_x, macula_y, manuel) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(oeil) DO UPDATE SET disque_x=excluded.disque_x, "
            "disque_y=excluded.disque_y, disque_rayon_px=excluded.disque_rayon_px, "
            "macula_x=excluded.macula_x, macula_y=excluded.macula_y, "
            "manuel=excluded.manuel",
            (
                r.oeil,
                r.disque_x,
                r.disque_y,
                r.disque_rayon_px,
                r.macula_x,
                r.macula_y,
                int(r.manuel),
            ),
        )
        self.cx.commit()

    def reperes(self, oeil: str) -> Reperes | None:
        l = self.cx.execute("SELECT * FROM reperes WHERE oeil = ?", (oeil,)).fetchone()
        if not l:
            return None
        return Reperes(
            oeil=l["oeil"],
            disque_x=l["disque_x"],
            disque_y=l["disque_y"],
            disque_rayon_px=l["disque_rayon_px"],
            macula_x=l["macula_x"],
            macula_y=l["macula_y"],
            manuel=bool(l["manuel"]),
        )

    # -- recalage ------------------------------------------------------------

    def enregistrer_recalage(self, q: QualiteRecalage, empreinte: str) -> None:
        self.cx.execute(
            "INSERT INTO recalage (cliche_id, empreinte, matrice, inliers, "
            "residu_rms, valide) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(cliche_id) DO UPDATE SET empreinte=excluded.empreinte, "
            "matrice=excluded.matrice, inliers=excluded.inliers, "
            "residu_rms=excluded.residu_rms, valide=excluded.valide",
            (
                q.cliche_id,
                empreinte,
                json.dumps(q.matrice),
                q.inliers,
                q.residu_rms,
                int(q.valide),
            ),
        )
        self.cx.commit()

    def recalage(self, cliche_id: int) -> QualiteRecalage | None:
        l = self.cx.execute(
            "SELECT * FROM recalage WHERE cliche_id = ?", (cliche_id,)
        ).fetchone()
        if not l:
            return None
        return QualiteRecalage(
            cliche_id=l["cliche_id"],
            matrice=json.loads(l["matrice"]),
            inliers=l["inliers"],
            residu_rms=l["residu_rms"],
            valide=bool(l["valide"]),
        )

    def recalages_invalides(self, oeil: str) -> list[tuple[int, str]]:
        """Cliches dont le recalage a echoue : (cliche_id, fichier)."""
        return [
            (l["cliche_id"], l["fichier"])
            for l in self.cx.execute(
                "SELECT r.cliche_id, c.fichier FROM recalage r "
                "JOIN cliche c ON c.id = r.cliche_id "
                "JOIN seance s ON s.id = c.seance_id "
                "WHERE s.oeil = ? AND r.valide = 0 ORDER BY c.fichier",
                (oeil,),
            )
        ]

    # -- analyses et mesures -------------------------------------------------

    def creer_analyse(
        self, oeil: str, methode: str, params_json: str, empreinte: str
    ) -> int:
        cur = self.cx.execute(
            "INSERT INTO analyse (oeil, methode, params_json, empreinte, cree_le) "
            "VALUES (?,?,?,?,?)",
            (
                oeil,
                methode,
                params_json,
                empreinte,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.cx.commit()
        return cur.lastrowid

    def derniere_analyse(self, oeil: str, methode: str | None = None) -> int | None:
        sql = "SELECT id FROM analyse WHERE oeil = ?"
        args: list = [oeil]
        if methode:
            sql += " AND methode = ?"
            args.append(methode)
        sql += " ORDER BY id DESC LIMIT 1"
        l = self.cx.execute(sql, args).fetchone()
        return l["id"] if l else None

    def enregistrer_mesure(
        self,
        analyse_id: int,
        seance_id: int,
        nombre: int,
        surface_px: int,
        surface_mm2: float | None,
        apparues: int = 0,
        disparues: int = 0,
        fusions: int = 0,
        scissions: int = 0,
    ) -> None:
        self.cx.execute(
            "INSERT OR REPLACE INTO mesure (analyse_id, seance_id, nombre, surface_px, "
            "surface_mm2, apparues, disparues, fusions, scissions) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                analyse_id,
                seance_id,
                nombre,
                surface_px,
                surface_mm2,
                apparues,
                disparues,
                fusions,
                scissions,
            ),
        )
        self.cx.commit()

    def mesures(self, analyse_id: int) -> list[dict]:
        return [
            dict(l)
            for l in self.cx.execute(
                "SELECT m.*, s.date AS date FROM mesure m "
                "JOIN seance s ON s.id = m.seance_id "
                "WHERE m.analyse_id = ? ORDER BY s.date",
                (analyse_id,),
            )
        ]

    def enregistrer_druses(
        self, analyse_id: int, seance_id: int, druses: list[Druse]
    ) -> None:
        self.cx.execute(
            "DELETE FROM druse WHERE analyse_id = ? AND seance_id = ?",
            (analyse_id, seance_id),
        )
        self.cx.executemany(
            "INSERT INTO druse (analyse_id, seance_id, piste_id, cx, cy, surface_px, "
            "excentricite, solidite, intensite_moy) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    analyse_id,
                    seance_id,
                    d.piste_id,
                    d.cx,
                    d.cy,
                    d.surface_px,
                    d.excentricite,
                    d.solidite,
                    d.intensite_moy,
                )
                for d in druses
            ],
        )
        self.cx.commit()

    def druses(self, analyse_id: int, seance_id: int | None = None) -> list[dict]:
        sql = "SELECT * FROM druse WHERE analyse_id = ?"
        args: list = [analyse_id]
        if seance_id is not None:
            sql += " AND seance_id = ?"
            args.append(seance_id)
        return [dict(l) for l in self.cx.execute(sql, args)]

    def piste(self, analyse_id: int, piste_id: int) -> list[dict]:
        """Historique individuel d'une druse, dans l'ordre chronologique."""
        return [
            dict(l)
            for l in self.cx.execute(
                "SELECT d.*, s.date AS date FROM druse d "
                "JOIN seance s ON s.id = d.seance_id "
                "WHERE d.analyse_id = ? AND d.piste_id = ? ORDER BY s.date",
                (analyse_id, piste_id),
            )
        ]

    # -- statistiques --------------------------------------------------------

    def resume(self) -> dict:
        res = {}
        for oeil in ("D", "G"):
            l = self.cx.execute(
                "SELECT COUNT(DISTINCT s.id) AS seances, COUNT(c.id) AS cliches "
                "FROM seance s LEFT JOIN cliche c ON c.seance_id = s.id "
                "WHERE s.oeil = ?",
                (oeil,),
            ).fetchone()
            bornes = self.cx.execute(
                "SELECT MIN(date) AS d0, MAX(date) AS d1 FROM seance WHERE oeil = ?",
                (oeil,),
            ).fetchone()
            res[oeil] = {
                "seances": l["seances"],
                "cliches": l["cliches"],
                "debut": bornes["d0"],
                "fin": bornes["d1"],
            }
        return res
