"""Interface en ligne de commande.

Permet de dérouler toute la chaine sans interface graphique - utile pour les
traitements longs, les scripts et les tests de non-regression.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import LIBELLE_OEIL, OEILS, Parametres, emplacements_par_defaut
from .pipeline import Pipeline
from .stockage.base import Base
from .stockage.bibliotheque import Bibliotheque


def _contexte(args) -> tuple[Bibliotheque, Base, Pipeline]:
    emp = emplacements_par_defaut(args.data)
    base = Base(emp.base)
    biblio = Bibliotheque(emp, base)
    params = Parametres()
    if getattr(args, "params", None):
        params = Parametres.depuis_json(Path(args.params).read_text())
    return biblio, base, Pipeline(biblio, base, params)


def _barre(etiquette: str, i: int, n: int, detail: str = "") -> None:
    """Barre de progression. Hors terminal, n'affiche que des jalons."""
    if not sys.stdout.isatty():
        pas = max(1, n // 10)
        if i == n or i % pas == 0:
            print(f"  {etiquette:16s} {i}/{n} {detail}", flush=True)
        return
    largeur = 28
    rempli = int(largeur * i / max(n, 1))
    barre = "#" * rempli + "." * (largeur - rempli)
    sys.stdout.write(f"\r  {etiquette:16s} [{barre}] {i}/{n} {detail[:24]:24s}")
    sys.stdout.flush()
    if i >= n:
        sys.stdout.write("\n")


# --- commandes ---------------------------------------------------------------


def cmd_importer(args) -> int:
    biblio, base, _ = _contexte(args)
    total = 0
    for source, oeil in args.sources:
        chemin = Path(source)
        if not chemin.exists():
            print(f"  introuvable : {chemin}", file=sys.stderr)
            continue
        print(f"Import de {chemin} ({LIBELLE_OEIL.get(oeil, oeil)})")
        res = biblio.importer_dossier(
            chemin,
            oeil=oeil,
            copier=not args.deplacer,
            rappel=lambda i, n, nom: _barre("import", i, n, nom),
        )
        print(f"  -> {res}")
        for ligne in res.ignores[:5]:
            print(f"     ignore : {ligne}")
        for ligne in res.erreurs[:5]:
            print(f"     ERREUR : {ligne}")
        total += res.importes
    print(f"\nTotal importe : {total}")
    cmd_resume(args)
    return 0


def cmd_resume(args) -> int:
    _, base, _ = _contexte(args)
    resume = base.resume()
    print("\nBibliotheque")
    for oeil in OEILS:
        r = resume[oeil]
        periode = f"{r['debut']} -> {r['fin']}" if r["debut"] else "vide"
        print(
            f"  {LIBELLE_OEIL[oeil]:12s} {r['seances']:4d} seance(s), "
            f"{r['cliches']:4d} cliche(s)   {periode}"
        )
    evenements = base.evenements()
    if evenements:
        print("\nEvenements")
        for e in evenements:
            print(f"  {e.date}  {e.libelle}")
    return 0


def cmd_evenement(args) -> int:
    _, base, _ = _contexte(args)
    jour = datetime.strptime(args.date, "%Y-%m-%d").date()
    base.ajouter_evenement(jour, args.libelle, args.note)
    print(f"Evenement ajoute : {jour} {args.libelle}")
    return 0


def cmd_normaliser(args) -> int:
    _, _, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        etat = pipeline.normaliser(oeil, forcer=args.forcer, rappel=_barre)
        print(f"  {etat}")
        for m in etat.messages[:12]:
            print(f"    {m}")
    return 0


def cmd_gabarit(args) -> int:
    _, _, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        gabarit = pipeline.construire_gabarit(oeil, forcer=args.forcer, rappel=_barre)
        if gabarit is None:
            print("  aucune image normalisee : lancer 'normaliser' d'abord")
        else:
            print(f"  gabarit : {pipeline.chemin_gabarit(oeil)}")
    return 0


def cmd_recaler(args) -> int:
    _, base, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        if pipeline.lire_gabarit(oeil) is None:
            pipeline.construire_gabarit(oeil, rappel=_barre)
        etat = pipeline.recaler(oeil, forcer=args.forcer, rappel=_barre)
        print(f"  {etat}")
        for m in etat.messages[:12]:
            print(f"    {m}")
        invalides = base.recalages_invalides(oeil)
        if invalides:
            print(f"  {len(invalides)} cliche(s) a revoir :")
            for _, fichier in invalides[:10]:
                print(f"    {fichier}")
    return 0


def cmd_planche(args) -> int:
    import cv2

    _, base, pipeline = _contexte(args)
    from .qualite import planche_contact

    for oeil in args.oeil:
        etape = args.etape
        images = []
        for seance in base.seances(oeil):
            if etape == "recale":
                img = pipeline.lire_recale(oeil, seance.date)
            else:
                img = pipeline.lire_normalise(oeil, seance.date)
            if img is not None:
                images.append((str(seance.date), img))
        if not images:
            print(f"{LIBELLE_OEIL[oeil]} : aucune image a l'etape '{etape}'")
            continue
        planche = planche_contact(images, largeur_vignette=args.vignette)
        cible = pipeline.emp.exports / f"planche_{etape}_{oeil}.jpg"
        cible.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cible), planche, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{LIBELLE_OEIL[oeil]} : {len(images)} vignettes -> {cible}")
    return 0


def cmd_tout(args) -> int:
    """Deroule la chaine complete."""
    cmd_normaliser(args)
    cmd_gabarit(args)
    cmd_recaler(args)
    return 0


# --- analyse ----------------------------------------------------------------


def cmd_detecter(args) -> int:
    from .analyse.execution import executer_detection

    _, base, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        resultat = executer_detection(pipeline, base, oeil, rappel=_barre)
        print(f"  {resultat}")
    return 0


def cmd_evolution(args) -> int:
    from .analyse.execution import executer_evolution

    _, base, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        resultat = executer_evolution(
            pipeline, base, oeil, methode=args.methode, rappel=_barre
        )
        print(f"  {resultat}")
        if args.csv:
            chemin = pipeline.emp.exports / f"evolution_{oeil}.csv"
            resultat.exporter_csv(chemin)
            print(f"  CSV -> {chemin}")
    return 0


def cmd_controle(args) -> int:
    """Contrôle negatif sur les paires rapprochees."""
    from .analyse.controle import controle_negatif, controle_sequence

    _, base, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        rapport = controle_negatif(
            pipeline, base, oeil, ecart_max_jours=args.jours, rappel=_barre
        )
        print(rapport.rendu_texte())

        sequences = controle_sequence(
            pipeline, base, oeil, ecart_max_jours=args.jours, rappel=_barre
        )
        if sequences:
            print("\n  Chaine complete sur les suites rapprochees")
            print("  (detection + appariement + regle de confirmation) :")
            for s in sequences:
                print(f"    {s}")
            total = sum(s.evenements_confirmes for s in sequences)
            eligibles = sum(s.dates_eligibles for s in sequences)
            print(
                f"\n  => {total} faux positif(s) sur {eligibles} date(s) "
                f"eligible(s), soit {total / max(eligibles, 1):.0f} par date."
            )
            print(
                "     Tout comptage d'evenements inferieur a ce seuil est "
                "du bruit de segmentation."
            )
        if args.csv:
            chemin = pipeline.emp.exports / f"controle_negatif_{oeil}.csv"
            rapport.exporter_csv(chemin)
            print(f"  CSV -> {chemin}")
    return 0


def cmd_video(args) -> int:
    from .video.timelapse import generer_timelapse
    from .video.mise_en_evidence import generer_mise_en_evidence

    _, base, pipeline = _contexte(args)
    for oeil in args.oeil:
        print(f"\n{LIBELLE_OEIL[oeil]}")
        if args.type in ("timelapse", "les-deux"):
            cible = pipeline.emp.exports / f"timelapse_{oeil}.mp4"
            generer_timelapse(pipeline, base, oeil, cible, rappel=_barre)
            print(f"  timelapse -> {cible}")
        if args.type in ("evolution", "les-deux"):
            cible = pipeline.emp.exports / f"evolution_{oeil}.mp4"
            generer_mise_en_evidence(pipeline, base, oeil, cible, rappel=_barre)
            print(f"  mise en evidence -> {cible}")
    return 0


def cmd_interface(args) -> int:
    from .ui.app import lancer

    return lancer(args.data)


# --- analyse des arguments ---------------------------------------------------


def _ajouter_oeil(sp) -> None:
    sp.add_argument(
        "--oeil",
        nargs="+",
        choices=list(OEILS),
        default=list(OEILS),
        help="oeil(s) a traiter (defaut : les deux)",
    )


def _paire_source(valeur: str) -> tuple[str, str]:
    """Analyse 'chemin:D' ou 'chemin' (oeil deduit du nom du dossier)."""
    if ":" in valeur:
        chemin, _, oeil = valeur.rpartition(":")
        if oeil.upper() in OEILS:
            return chemin, oeil.upper()
        valeur = f"{chemin}:{oeil}"
    nom = Path(valeur).name.lower()
    if "droit" in nom:
        return valeur, "D"
    if "gauche" in nom:
        return valeur, "G"
    return valeur, None  # type: ignore[return-value]


def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mama-eyes",
        description=(
            "Suivi photographique longitudinal du fond d'oeil. "
            "Outil personnel, sans finalite diagnostique."
        ),
    )
    p.add_argument("--data", default=None, help="racine de la bibliotheque (defaut : ./data)")
    p.add_argument("--params", default=None, help="fichier JSON de parametres")
    sous = p.add_subparsers(dest="commande", required=True)

    sp = sous.add_parser("importer", help="importer des cliches")
    sp.add_argument(
        "sources",
        nargs="+",
        type=_paire_source,
        help="dossier(s), eventuellement suffixes par :D ou :G",
    )
    sp.add_argument("--deplacer", action="store_true", help="deplacer au lieu de copier")
    sp.set_defaults(fonction=cmd_importer)

    sp = sous.add_parser("resume", help="etat de la bibliotheque")
    sp.set_defaults(fonction=cmd_resume)

    sp = sous.add_parser("evenement", help="ajouter un repere therapeutique")
    sp.add_argument("date", help="AAAA-MM-JJ")
    sp.add_argument("libelle")
    sp.add_argument("--note", default=None)
    sp.set_defaults(fonction=cmd_evenement)

    sp = sous.add_parser("normaliser", help="etape 1 : photometrie et fusion")
    _ajouter_oeil(sp)
    sp.add_argument("--forcer", action="store_true", help="ignorer le cache")
    sp.set_defaults(fonction=cmd_normaliser)

    sp = sous.add_parser("gabarit", help="construire le gabarit de groupe")
    _ajouter_oeil(sp)
    sp.add_argument("--forcer", action="store_true")
    sp.set_defaults(fonction=cmd_gabarit)

    sp = sous.add_parser("recaler", help="etape 2 : recalage sur le gabarit")
    _ajouter_oeil(sp)
    sp.add_argument("--forcer", action="store_true")
    sp.set_defaults(fonction=cmd_recaler)

    sp = sous.add_parser("tout", help="normaliser + gabarit + recaler")
    _ajouter_oeil(sp)
    sp.add_argument("--forcer", action="store_true")
    sp.set_defaults(fonction=cmd_tout)

    sp = sous.add_parser("planche", help="planche-contact de controle")
    _ajouter_oeil(sp)
    sp.add_argument("--etape", choices=["normalise", "recale"], default="recale")
    sp.add_argument("--vignette", type=int, default=260)
    sp.set_defaults(fonction=cmd_planche)

    sp = sous.add_parser("detecter", help="segmentation des druses")
    _ajouter_oeil(sp)
    sp.set_defaults(fonction=cmd_detecter)

    sp = sous.add_parser("evolution", help="suivi temporel et statistiques")
    _ajouter_oeil(sp)
    sp.add_argument(
        "--methode",
        choices=["absolue", "vs_precedent", "vs_reference"],
        default="absolue",
    )
    sp.add_argument("--csv", action="store_true")
    sp.set_defaults(fonction=cmd_evolution)

    sp = sous.add_parser(
        "controle", help="controle negatif sur les paires rapprochees"
    )
    _ajouter_oeil(sp)
    sp.add_argument("--jours", type=int, default=10, help="ecart maximal (defaut 10)")
    sp.add_argument("--csv", action="store_true")
    sp.set_defaults(fonction=cmd_controle)

    sp = sous.add_parser("video", help="generer les videos")
    _ajouter_oeil(sp)
    sp.add_argument(
        "--type", choices=["timelapse", "evolution", "les-deux"], default="les-deux"
    )
    sp.set_defaults(fonction=cmd_video)

    sp = sous.add_parser("interface", help="lancer l'interface graphique")
    sp.set_defaults(fonction=cmd_interface)

    return p


def main(argv: list[str] | None = None) -> int:
    args = construire_parseur().parse_args(argv)
    return args.fonction(args)


if __name__ == "__main__":
    raise SystemExit(main())
