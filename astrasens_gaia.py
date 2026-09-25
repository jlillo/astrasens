"""Optional catalogue lookup, isolated so the caller can enforce a deadline."""
import argparse
import contextlib
import json
import re
import sys


def lookup(gaia, simbad, catalogs, *, source_id=None, target=None, tic=None):
    def query(adql):
        return gaia.launch_job(adql).get_results()

    def identifier(value):
        text = str(value).strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError('Invalid catalogue identifier: '+text)
        return int(text)

    if source_id is None and tic is not None:
        print('Resolving TIC '+str(tic), file=sys.stderr, flush=True)
        table = catalogs.query_criteria(catalog='Tic', ID=identifier(tic))
        if table is None or not len(table):
            raise ValueError('TIC not found')
        dr2 = identifier(table['GAIA'][0])
        matches = query('SELECT n.dr3_source_id, g.phot_g_mean_mag '
                        'FROM gaiadr3.dr2_neighbourhood AS n '
                        'JOIN gaiadr3.gaia_source AS g ON g.source_id=n.dr3_source_id '
                        f'WHERE n.dr2_source_id={dr2} ORDER BY g.phot_g_mean_mag ASC')
        if not len(matches):
            raise ValueError('No Gaia DR3 counterpart for TIC')
        source_id = matches['dr3_source_id'][0]
        if len(matches) > 1:
            print('Multiple DR3 matches for TIC; selecting brightest.', file=sys.stderr, flush=True)
    if source_id is None:
        if not target or not target.strip():
            raise ValueError('No target name, TIC or Gaia DR3 identifier supplied')
        target = target.replace('_', ' ').strip()
        print('Resolving target in SIMBAD: '+target, file=sys.stderr, flush=True)
        ids = simbad.query_objectids(target)
        if ids is not None and len(ids):
            column = next(c for c in ids.colnames if c.lower() == 'id')
            for value in ids[column]:
                if isinstance(value, bytes):
                    value = value.decode()
                match = re.fullmatch(r'Gaia DR3\s+(\d+)', str(value).strip())
                if match:
                    source_id = match.group(1)
                    break
        if source_id is None:
            raise ValueError('No Gaia DR3 identifier found in SIMBAD for '+target)
    source_id = identifier(source_id)
    print('Querying Gaia DR3 position: '+str(source_id), file=sys.stderr, flush=True)
    primary = query('SELECT source_id, ra, dec FROM gaiadr3.gaia_source '
                    f'WHERE source_id={source_id}')
    if not len(primary):
        raise ValueError('Gaia DR3 source not found')
    ra, dec = float(primary['ra'][0]), float(primary['dec'][0])
    print('Querying Gaia neighbours within 5 arcsec', file=sys.stderr, flush=True)
    # Synchronous cone query avoids server-side async polling; the parent bounds
    # the total runtime, including resolution, imports and all HTTP requests.
    neighbours = query('SELECT source_id, ra, dec FROM gaiadr3.gaia_source '
                       "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                       f"CIRCLE('ICRS', {ra:.12f}, {dec:.12f}, {5/3600:.15f}))")
    dx, dy, ids = [], [], []
    for row in neighbours:
        sid = int(row['source_id'])
        if sid == source_id:
            continue
        # Preserve the existing plot's RA/Dec coordinate convention.
        dx.append(-(float(row['ra'])-ra)*3600)
        dy.append((float(row['dec'])-dec)*3600)
        ids.append(sid)
    return dict(delta_ra=dx, delta_dec=dy, gid=ids, ngaia=len(ids)+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-id')
    parser.add_argument('--target')
    parser.add_argument('--tic')
    args = parser.parse_args()
    try:
        # Astroquery may print service notices at import time. Keep stdout JSON.
        with contextlib.redirect_stdout(sys.stderr):
            from astroquery.gaia import Gaia
            from astroquery.simbad import Simbad
            from astroquery.mast import Catalogs
            result = lookup(Gaia, Simbad, Catalogs, source_id=args.source_id,
                            target=args.target, tic=args.tic)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
