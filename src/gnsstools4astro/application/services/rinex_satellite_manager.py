"""\
Módulo: packages/gnsstools4astro/src/gnsstools4astro/application/services/rinex_satellite_manager.py

Descrição:
    Facade de alto nível (UX) para satélites CRIADOS A PARTIR DE RINEX (NAV).

    Lê arquivos RINEX NAV (texto) ou datasets NAV já parseados (ex.: cache
    NetCDF do legado), extrai as efemérides de transmissão (broadcast) e
    cria os satélites GNSS (SV/PRN/constelação). Constrói tracks
    (alt/az/RA/Dec) no referencial de uma estação e delega a visualização
    (deferida) ao ``radioskysims-viz``.

    Exemplo (notebook):
        >>> mgr = RinexSatelliteManager.from_nav("braz1621.24n")
        >>> mgr.list_satellites()
        >>> ds = mgr.build_tracks()            # xr.Dataset (sv, time)
        >>> mgr.view_mollweide()               # RA/Dec (lazy viz)

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from gnsstools4astro.domain.rinex import RinexFileType, RinexRequest, RinexSource
from gnsstools4astro.domain.satellite import (
    BroadcastEphemeris,
    Satellite,
    constellation_from_sv,
)
from gnsstools4astro.domain.station import DEFAULT_STATION_POSITION, GNSSStation
from gnsstools4astro.infrastructure.ephemeris.broadcast import propagate_kepler
from gnsstools4astro.infrastructure.ephemeris.skyfield import SkyfieldTracks
from gnsstools4astro.infrastructure.rinex.reader import RinexReader

__all__ = ["RinexAssembled", "RinexSatelliteManager"]

_TRACK_VARS = [
    "alt_deg",
    "az_deg",
    "ra_deg",
    "dec_deg",
    "distance_km",
]


def _unique_satellites(ephemerides: list[BroadcastEphemeris]) -> list[Satellite]:
    """Satélites únicos (por SV) a partir das efemérides."""
    seen: dict[str, Satellite] = {}
    for eph in ephemerides:
        if eph.sv not in seen:
            seen[eph.sv] = Satellite(
                sv=eph.sv, constellation=constellation_from_sv(eph.sv)
            )
    return list(seen.values())


def _time_grid(
    start: datetime,
    end: datetime,
    step_s: float,
) -> list[datetime]:
    """Grade de tempo (UTC) com passo fixo em segundos."""
    step = timedelta(seconds=step_s)
    times: list[datetime] = []
    t = start
    while t <= end:
        times.append(t)
        t += step
    return times


def _to_naive(t) -> datetime:
    """Converte ``np.datetime64`` (UTC) → ``datetime`` naive, sem aviso de ns."""
    return pd.Timestamp(np.datetime64(t, "s")).to_pydatetime()


class RinexAssembled:
    """Montado de satélites RINEX (tracks + signals) — API compatível com o
    ``SatelliteAssembled`` do TLE (``.tracks``, ``.signals``, ``plot_tracks``,
    ``plot_signals``, ``summary``).
    """

    def __init__(
        self,
        tracks: xr.Dataset,
        signals: xr.Dataset,
        sv_ids: list[str],
    ) -> None:
        self.tracks = tracks
        self.signals = signals
        self.sv_ids = list(sv_ids)

    def plot_tracks(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """Mapa Mollweide RA/Dec dos satélites (sem continentes — céu)."""
        from radioskysims_viz.satellite_rinex import rinex_tracks_mollweide

        return rinex_tracks_mollweide(self.tracks, title=title, **kwargs)

    def plot_signals(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """PSD (dB W/Hz) × frequência — referência de sinal por constelação."""
        from radioskysims_viz.satellite_rinex import rinex_signals_plot

        return rinex_signals_plot(self.signals, title=title, **kwargs)

    def summary(self) -> str:
        n_track = self.tracks.sizes.get("sv", 0)
        n_sig = self.signals.sizes.get("sv", 0)
        return (
            f"RinexAssembled({n_track} satélites; tracks {dict(self.tracks.sizes)}, "
            f"signals {n_sig})"
        )

    def __repr__(self) -> str:
        return self.summary()


class RinexSatelliteManager:
    """Satélites criados a partir de RINEX NAV (efemérides de transmissão).

    Attributes:
        satellites (list[Satellite]): Satélites únicos (SV/PRN/constelação).
        ephemerides (list[BroadcastEphemeris]): Efemérides de transmissão.
        nav (xr.Dataset | None): Dataset NAV de origem (se disponível).
    """

    def __init__(
        self,
        satellites: list[Satellite],
        ephemerides: list[BroadcastEphemeris],
        nav: xr.Dataset | None = None,
    ) -> None:
        self.satellites = list(satellites)
        self.ephemerides = list(ephemerides)
        self.nav = nav
        self._tracks: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    @classmethod
    def from_nav(cls, path: str | Path) -> "RinexSatelliteManager":
        """Carrega um arquivo RINEX NAV (texto) e cria os satélites."""
        nav = RinexReader().load_nav(path)
        return cls.from_nav_dataset(nav)

    @classmethod
    def from_nav_dataset(cls, nav: xr.Dataset) -> "RinexSatelliteManager":
        """Cria os satélites a partir de um dataset NAV já parseado.

        Aceita tanto o resultado de ``RinexReader.load_nav`` quanto caches
        NetCDF com o mesmo esquema (coords ``time``/``sv``).
        """
        ephemerides = RinexReader().to_ephemerides(nav)
        return cls(
            satellites=_unique_satellites(ephemerides),
            ephemerides=ephemerides,
            nav=nav,
        )

    @classmethod
    def for_station(
        cls,
        station: str,
        *,
        ref_date: datetime,
        cache_dir: str | Path | None = None,
        source: Any = None,
    ) -> "RinexSatelliteManager":
        """Manager configurado para uma estação/data — os dados (cache ou
        download RINEX NAV) são carregados sob demanda no ``build``.
        """
        mgr = cls(satellites=[], ephemerides=[], nav=None)
        mgr._station = station
        mgr._ref_date = ref_date
        mgr._cache_dir = Path(cache_dir or "data/rinex")
        mgr._source = source or RinexSource.RBMC
        return mgr

    def _ensure_station(self) -> None:
        """Valida que o manager foi configurado com uma estação (for_station)."""
        if not getattr(self, "_station", None):
            msg = (
                "RinexSatelliteManager sem estação — use for_station() ou "
                "from_nav()/from_nav_dataset()."
            )
            raise RuntimeError(msg)

    def _download(self, file_types: tuple[RinexFileType, ...], label: str, doy: str) -> Path:
        """Baixa online os tipos pedidos e devolve o primeiro arquivo do dia."""
        from gnsstools4astro.bootstrap.composition import build_download_use_case

        print(f"[online] baixando RINEX {label} ({self._station}) ...")
        req = RinexRequest(
            station=self._station,
            start=self._ref_date,
            end=self._ref_date + timedelta(hours=24),
            source=self._source,
            file_types=file_types,
        )
        res = build_download_use_case(self._cache_dir).run(req, self._cache_dir)
        files = [p for p in res.files.get(file_types[0], []) if doy in p.stem]
        if not files:
            msg = (
                f"download RINEX {label} não retornou arquivo para "
                f"{self._ref_date:%Y-%m-%d}"
            )
            raise RuntimeError(msg)
        return files[0]

    def download_nav(self, *, force: bool = False) -> Path:
        """Garante o NAV RINEX (cache local ou download online) da estação/data.

        Define ``satellites``/``ephemerides``/``nav`` a partir do arquivo.

        Returns
        -------
        Path
            Caminho do NAV local.
        """
        self._ensure_station()
        cache_dir = self._cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        doy = self._ref_date.strftime("%j")
        yy = self._ref_date.strftime("%y")
        matches = sorted(cache_dir.glob(f"{self._station}*{doy}*.{yy}n"))
        if matches and not force:
            nav_path = matches[0]
            print(f"[cache] RINEX NAV ({self._station}): {nav_path}")
        else:
            nav_path = self._download((RinexFileType.NAV,), "NAV", doy)
        loaded = type(self).from_nav(nav_path)
        self.satellites = loaded.satellites
        self.ephemerides = loaded.ephemerides
        self.nav = loaded.nav
        return nav_path

    def download_obs(self, *, force: bool = False) -> Path | None:
        """Garante o RINEX obs (cache local ou download online) da estação/data.

        O obs é usado pelo link budget. Retorna ``None`` se indisponível para
        a data (não levanta).
        """
        self._ensure_station()
        cache_dir = self._cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        doy = self._ref_date.strftime("%j")
        yy = self._ref_date.strftime("%y")
        matches = sorted(cache_dir.glob(f"{self._station}*{doy}*.{yy}[od]"))
        if matches and not force:
            print(f"[cache] RINEX OBS ({self._station}): {matches[0]}")
            return matches[0]
        try:
            return self._download((RinexFileType.OBS,), "OBS", doy)
        except Exception as exc:  # noqa: BLE001 — obs pode não existir p/ a data
            print(f"[aviso] RINEX OBS indisponível ({self._station}): {exc}")
            return None

    def _ensure_data(self) -> None:
        """Garante os dados: usa cache local ou baixa o NAV RINEX online."""
        if self.satellites:
            return
        self.download_nav()

    def build(self, time_range: Any, frequency_range: Any) -> RinexAssembled:
        """Monta o ``RinexAssembled`` (tracks + signals) — API compatível com
        ``SatelliteManager.build(TR, FR)``.

        Garante os dados (cache local ou download RINEX NAV online) antes de
        calcular tracks e sinais.
        """
        self._ensure_data()
        tracks = self.build_tracks(time_range=time_range)
        signals = self._build_signals(frequency_range)
        return RinexAssembled(
            tracks=tracks,
            signals=signals,
            sv_ids=list(self._best_ephemeris_per_sv()),
        )

    def _build_signals(
        self,
        frequency_range: Any = None,
        *,
        eirp_dbw_by_sv: dict[str, float] | None = None,
    ) -> xr.Dataset:
        """Dataset de sinais (sv, freq) — PSD analítica por banda, no grid do FR.

        Usa o MESMO código das PSDs analíticas que o TLE
        (``gnsstools4astro.domain.analytic_psd``): para cada banda da
        constelação dentro do ``frequency_range``, avalia a forma espectral
        closed-form (BPSK/BOC/CBOC) sobre o grid de frequências do FR.

        A potência por banda vem de ``eirp_dbw_by_sv`` (ex.: link budget do
        RINEX obs) ou da EIRP de referência (``signals.json``).

        Estrutura compatível com ``SatelliteAssembled.signals``
        (dims ``satellite``/``freq``, variável ``PSD`` em dBW/Hz).
        """
        from gnsstools4astro.domain.analytic_psd import (
            LOW_FLOOR_DB,
            analytic_psd,
            psd_db,
            psd_w_per_hz,
        )
        from gnsstools4astro.domain.constellations import get_constellation_signals

        ref = get_constellation_signals()

        # grid de frequências (MHz) do FR
        if hasattr(frequency_range, "range"):
            freqs_mhz = np.asarray(frequency_range.range, dtype=float)
        elif hasattr(frequency_range, "min_mhz"):
            freqs_mhz = np.arange(
                float(frequency_range.min_mhz),
                float(frequency_range.max_mhz) + 1e-9,
                float(frequency_range.resolution_mhz),
            )
        else:
            freqs_mhz = np.asarray(frequency_range, dtype=float)
        freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        freq_grid_hz = freqs_mhz * 1e6

        best = self._best_ephemeris_per_sv()
        svs = list(best)
        if not svs or freqs_mhz.size == 0:
            return xr.Dataset(
                coords={
                    "sv": svs,
                    "freq": np.array([], dtype=float),
                    "constellation": np.array([], dtype=str),
                }
            )

        psd = np.full((len(svs), freqs_mhz.size), LOW_FLOOR_DB)
        constellations: list[str] = []
        for i, sv in enumerate(svs):
            cons = constellation_from_sv(sv)
            cs = ref.get(cons)
            constellations.append(cons.value)
            if cs is None:
                continue
            if eirp_dbw_by_sv and sv in eirp_dbw_by_sv:
                power_w = 10.0 ** (float(eirp_dbw_by_sv[sv]) / 10.0)
            else:
                power_w = 10.0 ** (cs.eirp_dbw / 10.0)
            for band in cs.signals:
                f_mhz = band.frequency_hz / 1e6
                if not (freqs_mhz[0] - 1e-6 <= f_mhz <= freqs_mhz[-1] + 1e-6):
                    continue
                modulation = band.modulation or "BPSK(1)"
                norm = analytic_psd(modulation, freq_grid_hz, band.frequency_hz)
                band_db = psd_db(psd_w_per_hz(norm, power_w))
                psd[i, :] = np.maximum(psd[i, :], band_db)

        return xr.Dataset(
            {"PSD": (("sv", "freq"), psd)},
            coords={
                "sv": svs,
                "freq": freqs_mhz,
                "constellation": ("sv", constellations),
            },
            attrs={"provider": "rinex-nav-analytic-psd"},
        )

    def estimate_link_budget(
        self,
        obs: Any,
        *,
        receiver_gain_dbi: float = 0.0,
        t_noise_k: float = 290.0,
    ) -> list[Any]:
        """Estima EIRP por (SV × banda) a partir do RINEX obs + tracks (nav).

        Link budget (Friis): C/N0 do obs → potência recebida → EIRP, usando a
        distância dos tracks (nav). Alimenta ``signals.json`` via
        ``EstimateLinkBudgetUseCase.to_signals_reference``.

        Parameters
        ----------
        obs : str | Path | xr.Dataset
            Arquivo RINEX de observação (ou dataset já carregado).
        receiver_gain_dbi : float
            Ganho da antena receptora (dBi), padrão 0.
        t_noise_k : float
            Temperatura de ruído de referência (K), padrão 290.

        Returns
        -------
        list[LinkBudgetEstimate]
        """
        from gnsstools4astro.application.use_cases import (
            EstimateLinkBudgetUseCase,
            range_means_from_tracks,
        )
        from gnsstools4astro.domain.constellations import get_constellation_signals
        from gnsstools4astro.infrastructure.rinex.reader import RinexReader

        reader = RinexReader()
        if isinstance(obs, (str, Path)):
            obs = reader.load_obs(obs)
        cn0_per_sv = reader.mean_cn0_per_sv(obs)
        ranges = range_means_from_tracks(self._view_tracks())
        use_case = EstimateLinkBudgetUseCase(
            get_constellation_signals(),
            receiver_gain_dbi=receiver_gain_dbi,
            t_noise_k=t_noise_k,
        )
        return use_case.estimate(cn0_per_sv, ranges)

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def list_satellites(self) -> pd.DataFrame:
        """Tabela dos satélites (sv, constelação, PRN)."""
        return pd.DataFrame(
            [
                {
                    "sv": s.sv,
                    "constellation": s.constellation.value,
                    "prn": s.prn,
                }
                for s in self.satellites
            ]
        )

    def list_ephemerides(self) -> pd.DataFrame:
        """Resumo das efemérides (uma linha por SV, última época válida)."""
        best = self._best_ephemeris_per_sv()
        return pd.DataFrame(
            [
                {
                    "sv": eph.sv,
                    "constellation": eph.constellation.value,
                    "toc": eph.toc,
                    "toe": eph.toe,
                    "e": eph.e,
                    "sqrt_a": eph.sqrt_a,
                    "i0_deg": float(np.degrees(eph.i0)),
                    "omega0_deg": float(np.degrees(eph.omega0)),
                    "m0_deg": float(np.degrees(eph.m0)),
                }
                for eph in best.values()
            ]
        )

    def _best_ephemeris_per_sv(self) -> dict[str, BroadcastEphemeris]:
        """Melhor efeméride por SV (época mais recente)."""
        best: dict[str, BroadcastEphemeris] = {}
        for eph in self.ephemerides:
            cur = best.get(eph.sv)
            if cur is None or eph.toe > cur.toe:
                best[eph.sv] = eph
        return best

    # ------------------------------------------------------------------
    # Tracks
    # ------------------------------------------------------------------
    def build_tracks(
        self,
        station: GNSSStation | None = None,
        *,
        time_range: Any = None,
        times: list[datetime] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        step_s: float = 1200.0,
        min_alt_deg: float = 0.0,
    ) -> xr.Dataset:
        """Constrói tracks (alt/az/RA/Dec) dos satélites num intervalo.

        Propaga cada efeméride (ICD-GPS-200) e converte para o céu no
        referencial da estação via Skyfield. O resultado é cacheado em
        ``self._tracks`` (as views usam este mesmo dataset).

        Args:
            time_range: ``TimeRange``-like (com ``to_datetime64()``, ex.: o
                ``TimeRange`` de referência usado pelo TLE) OU tupla
                ``(start, end, step_s)``. Sobrepõe ``times``/``start``/``end``.
            times: Grade de tempo explícita (naive UTC).
            start, end, step_s: Janela alternativa (default 6 h centrada nas
                épocas das efemérides).

        Returns:
            xr.Dataset com coords ``(sv, time)`` e variáveis
            ``alt_deg, az_deg, ra_deg, dec_deg, distance_km``.
        """
        station = station or DEFAULT_STATION_POSITION
        best = self._best_ephemeris_per_sv()
        if not best:
            ds = xr.Dataset(
                coords={"sv": [], "time": np.array([], dtype="datetime64[ns]")}
            )
            self._tracks = ds
            return ds

        if time_range is not None:
            # ``TimeRange``-like: usa a grade de tempo dele (UTC naive).
            try:
                t64 = np.asarray(time_range.to_datetime64()).astype("datetime64[ns]")
                times = [_to_naive(t) for t in t64]
            except AttributeError:
                # tupla (start, end, step_s)
                start, end, step_s = time_range
        if times is None:
            toe_list = [e.toe for e in best.values()]
            # Janela padrão de 6h centrada nas épocas das efemérides (as
            # melhores por SV costumam compartilhar o mesmo ``toe``; ancorar
            # só no mínimo colapsaria a grade para poucos pontos).
            center = min(toe_list) + (max(toe_list) - min(toe_list)) / 2.0
            t_start = start or (center - timedelta(hours=3))
            t_end = end or (center + timedelta(hours=3))
            times = _time_grid(t_start, t_end, step_s)
        if not times:
            ds = xr.Dataset(
                coords={"sv": [], "time": np.array([], dtype="datetime64[ns]")}
            )
            self._tracks = ds
            return ds

        svs = list(best)
        n_sv, n_t = len(svs), len(times)
        data = {v: np.full((n_sv, n_t), np.nan) for v in _TRACK_VARS}
        sky = SkyfieldTracks(station)

        for i, sv in enumerate(svs):
            eph = best[sv]
            for j, t in enumerate(times):
                try:
                    # ``propagate_kepler`` com um único datetime já retorna a
                    # posição ECEF (3,); NÃO indexar [0] (pegaria só a componente x).
                    pos = lambda tt, _eph=eph: propagate_kepler(_eph, tt)  # noqa: E731
                    sp = sky.ecef_to_sky(pos, t)
                except Exception:  # noqa: BLE001 - propaga NaN p/ SV problemático
                    continue
                data["alt_deg"][i, j] = sp.alt_deg
                data["az_deg"][i, j] = sp.az_deg
                data["ra_deg"][i, j] = sp.ra_deg
                data["dec_deg"][i, j] = sp.dec_deg
                data["distance_km"][i, j] = sp.distance_km

        if min_alt_deg > 0:
            below = data["alt_deg"] < min_alt_deg
            for v in _TRACK_VARS:
                data[v][below] = np.nan

        ds = xr.Dataset(
            data_vars={v: (("sv", "time"), data[v]) for v in _TRACK_VARS},
            coords={
                "sv": svs,
                "time": np.asarray(
                    [
                        t.astimezone(UTC).replace(tzinfo=None)
                        if t.tzinfo is not None
                        else t
                        for t in times
                    ],
                    dtype="datetime64[ns]",
                ),
            },
            attrs={
                "provider": "rinex-nav",
                "station": station.name,
                "n_satellites": len(svs),
            },
        )
        self._tracks = ds
        return ds

    # ------------------------------------------------------------------
    # Visualização (deferida ao radioskysims-viz)
    # ------------------------------------------------------------------
    def _view_tracks(self):
        """Dataset cacheado por ``build_tracks`` (constrói na hora se vazio)."""
        if self._tracks is None:
            self._tracks = self.build_tracks()
        return self._tracks

    def view_mollweide(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """RA/Dec dos satélites RINEX no mapa Mollweide (lazy viz)."""
        from radioskysims_viz.satellite_rinex import rinex_tracks_mollweide

        return rinex_tracks_mollweide(self._view_tracks(), title=title, **kwargs)

    def view_skychart(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """Carta local ALT/AZ dos satélites RINEX (lazy viz)."""
        from radioskysims_viz.satellite_rinex import rinex_tracks_skychart

        return rinex_tracks_skychart(self._view_tracks(), title=title, **kwargs)

    # ------------------------------------------------------------------
    # Introspect
    # ------------------------------------------------------------------
    def summary(self) -> str:
        df = self.list_satellites()
        counts = df["constellation"].value_counts().to_dict()
        return (
            f"RinexSatelliteManager — {len(self.satellites)} satélites "
            f"(constelações: {counts})"
        )

    def __len__(self) -> int:
        return len(self.satellites)

    def __repr__(self) -> str:
        return f"<RinexSatelliteManager: {len(self.satellites)} satélites (RINEX NAV)>"
