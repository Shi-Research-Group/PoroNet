def atoms_from_mofdb_df(mofdb_df):
    """
    This function takes a MOFDB dataframe from get_data_from_mofdb routine, which has a column called 'cif' that contains CIF data as strings, and
    returns a list of ASE atoms objects

    :param mofdb_df: the dataframe containing the MOFDB data
    :return: A list of atoms objects.
    """
    import numpy as np
    from tqdm import tqdm

    # import dask.bag as db
    # from dask.diagnostics import ProgressBar
    # with ProgressBar():
    #     atoms_list = db.from_sequence(mofdb_df['cif'], npartitions=20).map(read_cif_string).compute()
    # from rich.progress import track
    atoms_list = [read_cif_string(x) for x in tqdm(mofdb_df["cif"])]
    for i, a in enumerate(atoms_list):
        if a != "NaN":
            a.info["name"] = mofdb_df["name"][i]

    return np.array(atoms_list, dtype="object")

def read_cif_string(cif):
    """
    It takes a string of CIF data and returns an ASE atoms object.

    :param cif: The CIF file as a string
    :return: the ASE atoms object.
    """
    from ase.io import read
    import io

    try:
        output = read(io.BytesIO(bytes(cif, encoding="utf-8")), format="cif")
    except:
        output = "NaN"
    return output

global forcefields
forcefields = dict(
    uff={
        "Ac": [16.60, 3.10],
        "Ag": [18.11, 2.80],
        "Al": [254.09, 4.01],
        "Am": [7.04, 3.01],
        "Ar": [93.08, 3.45],
        "As": [155.47, 3.77],
        "At": [142.89, 4.23],
        "Au": [19.62, 2.93],
        "B": [90.57, 3.64],
        "Ba": [183.15, 3.30],
        "Be": [42.77, 2.45],
        "Bi": [260.63, 3.89],
        "Bk": [6.54, 2.97],
        "Br": [126.29, 3.73],
        "C": [52.83, 3.43],
        "Ca": [119.75, 3.03],
        "Cd": [114.72, 2.54],
        "Ce": [6.54, 3.17],
        "Cf": [6.54, 2.95],
        "Cl": [114.21, 3.52],
        "Cm": [6.54, 2.96],
        "Co": [7.04, 2.56],
        "Cr": [7.55, 2.69],
        "Cu": [2.52, 3.11],
        "Cs": [22.64, 4.02],
        "Dy": [3.52, 3.05],
        "Eu": [4.03, 3.11],
        "Er": [3.52, 3.02],
        "Es": [6.04, 2.94],
        "F": [25.16, 3.00],
        "Fe": [6.54, 2.59],
        "Fm": [6.04, 2.93],
        "Fr": [25.16, 4.37],
        "Ga": [208.81, 3.90],
        "Ge": [190.69, 3.81],
        "Gd": [4.53, 3.00],
        "H": [22.14, 2.57],
        "He": [10.9, 2.64],
        "Hf": [36.23, 2.80],
        "Hg": [193.71, 2.41],
        "Ho": [3.52, 3.04],
        "I": [170.57, 4.01],
        "In": [301.39, 3.98],
        "Ir": [36.73, 2.53],
        "K": [17.61, 3.40],
        "Kr": [110.69, 3.69],
        "La": [8.55, 3.14],
        "Li": [12.58, 2.18],
        "Lu": [20.63, 3.24],
        "Lr": [5.53, 2.88],
        "Md": [5.53, 2.92],
        "Mg": [55.85, 2.69],
        "Mn": [6.54, 2.64],
        "Mo": [28.18, 2.72],
        "N": [34.72, 3.26],
        "Na": [15.09, 2.66],
        "Ne": [21.13, 2.66],
        "Nb": [29.69, 2.82],
        "Nd": [5.03, 3.18],
        "No": [5.53, 2.89],
        "Ni": [7.55, 2.52],
        "Np": [9.56, 3.05],
        "O": [30.19, 3.12],
        "Os": [18.62, 2.78],
        "P": [153.46, 3.69],
        "Pa": [11.07, 3.05],
        "Pb": [333.59, 3.83],
        "Pd": [24.15, 2.58],
        "Pm": [4.53, 3.16],
        "Po": [163.52, 4.20],
        "Pr": [5.03, 3.21],
        "Pt": [40.25, 2.45],
        "Pu": [8.05, 3.05],
        "Ra": [203.27, 3.28],
        "Rb": [20.13, 3.67],
        "Re": [33.21, 2.63],
        "Rh": [26.67, 2.61],
        "Rn": [124.78, 4.25],
        "Ru": [28.18, 2.64],
        "S": [137.86, 3.59],
        "Sb": [225.91, 3.94],
        "Sc": [9.56, 2.94],
        "Se": [146.42, 3.75],
        "Si": [202.27, 3.83],
        "Sm": [4.03, 3.14],
        "Sn": [285.28, 3.91],
        "Sr": [118.24, 3.24],
        "Ta": [40.75, 2.82],
        "Tb": [3.52, 3.07],
        "Tc": [24.15, 2.67],
        "Te": [200.25, 3.98],
        "Th": [13.08, 3.03],
        "Ti": [8.55, 2.83],
        "Tl": [342.14, 3.87],
        "Tm": [3.02, 3.01],
        "U": [11.07, 3.02],
        "V": [8.05, 2.80],
        "W": [33.71, 2.73],
        "Xe": [167.04, 3.92],
        "Y": [36.23, 2.98],
        "Yb": [114.72, 2.99],
        "Zn": [62.39, 2.46],
        "Zr": [34.72, 2.78],
        "CH4_sp3": [148.0, 3.73],
        "CH3_sp3": [98.0, 3.75],
        "CH2_sp3": [46.0, 3.95],
        "CH_sp3": [10, 4.68],
        "C_sp3": [0.5, 6.4],
        "H_h2": [0.0, 0.0],
        "H_com": [36.7, 2.958],
        "N_n2": [36.0, 3.31],
        "N_com": [0.0, 0.0],
        "O_co2": [79.0, 3.05],
        "C_co2": [27.0, 2.80],
        "Ow": [78.0, 3.15],
    }
)

def mix_lorentz_berthelot(ase_atoms, forcefield=forcefields["uff"], probe_symbol="He"):
    """
    It takes an ASE atoms object and a forcefield dictionary and returns a mixed forcefield dictionary

    :param ase_atoms: the atoms object from ASE
    :param forcefield: a dictionary of force field parameters for the atoms in the framework
    :param probe_symbol: the symbol of the probe atom, defaults to He (optional)
    :return: The mixed sigma and epsilon values for the interactions.
    """

    # * let 's ge tteh lorentz berthelot rules mixed force field params for the interactions
    import numpy as np
    import operator

    ff = forcefield
    frame_ff = np.vstack(
        operator.itemgetter(*ase_atoms.get_chemical_symbols())(ff)
    )  # framework force field
    probe_ff = ff[probe_symbol]  # proble
    lb_sigma = (frame_ff[:, 1] + probe_ff[1]) / 2  # mixed sigma
    lb_epsilon = np.sqrt(frame_ff[:, 0] * probe_ff[0])  # mixed epsilon
    lb_ff = np.vstack((lb_epsilon, lb_sigma)).T  # mixed sigma an d epsilon
    return lb_ff

def read_raspa_pdb(path_to_file, symbol_map=None):
    """
            created by Arun Gopalan, Snurr Research Group.                   _                                 ___  ___  ___ 
            _ __ ___  __ _  __| |  _ __ __ _ ___ _ __   __ _    / _ \/   \/ __\
            | '__/ _ \/ _` |/ _` | | '__/ _` / __| '_ \ / _` |  / /_)/ /\ /__\//
            | | |  __/ (_| | (_| | | | | (_| \__ \ |_) | (_| | / ___/ /_// \/  \
            |_|  \___|\__,_|\__,_| |_|  \__,_|___/ .__/ \__,_| \/  /___,'\_____/
                                                |_|                            

            Reads the output PDB movie file from RASPA into separate configurations, including the chemical symbols and
            cell dimensions. 
            :type path_to_file: str
            :param path_to_file: path to the RASPA PDB file
            :param symbol_map: Optional dictionary mapping custom symbols to standard element symbols.
        
            :raises:
            Not sure.
        
            :rtype: a python dictionary with 'cells', 'symbols' and 'coord' which stand for cell dimensions, printed symbols and coordinates
            """

    import numpy as np

    f = open(path_to_file).readlines()
    start = np.where(["MODEL" in line for line in f])[0] + 2  # * Start of config
    ends = np.where(["ENDMDL" in line for line in f])[0]  # End for config
    cryst = np.where(["CRYST" in line for line in f])[0]  # box shape for the config

    data = [f[start[i]: ends[i]] for i in range(len(start))]

    # Initialize lists to hold coordinates and symbols
    coords = []
    symbols = []  # Change: Initialize symbols as a list of lists
    skipped_lines = []  # List to hold skipped lines

    for d in data:
        coord = []
        config_symbols = []  # New list to hold symbols for the current configuration
        for line in d:
            parts = line.split()
            if len(parts) >= 7:  # Ensure there are enough parts to avoid index errors
                try:
                    coord.append(np.array(parts[4:7], dtype=float))
                    # Use symbol_map if provided, otherwise use the original symbol
                    symbol = parts[2]
                    if symbol_map and symbol in symbol_map:
                        symbol = symbol_map[symbol]  # Replace with mapped symbol if exists
                    config_symbols.append(symbol)  # Append the standardized symbol to the current config
                except ValueError:
                    skipped_lines.append(line.strip())  # Add the skipped line to the list
                    continue  # Skip lines that cannot be converted to float

        coords.append(np.array(coord))
        symbols.append(config_symbols)  # Change: Append the list of symbols for the current config

    cell_dims = np.array(
        [np.array(line.split())[1:].astype(float) for line in f if "CRYST" in line]
    )

    output = {
        "cells": cell_dims,
        "coords": np.array(coords, dtype=object),  # Use dtype=object to handle varying lengths
        "symbols": np.array(symbols, dtype=object)  # Use dtype=object for symbols
    }

    # Print warning if any lines were skipped
    if skipped_lines:
        print("Warning: The following lines were skipped due to conversion issues:")
        for skipped in skipped_lines:
            print(f"  - {skipped}")

    return output

async def fetch(url, session):
    async with session.get(url) as response:
        # print(await response.read())
        return await response.json()

async def bound_fetch(sem, url, session):
    # Getter function with semaphoe.
    async with sem:
        return await fetch(url, session)

async def get_mofdb_data(session, url, params, headers, npages, limit=1000):
    """Compile a asyncio Future containing the results of a search by making requests to MOFDb in parallel asynchronously

    :param session: object of the ClientSession
    :type session: aiohttp.ClientSession
    :param url: URL of the website. Designed for MOFDb, so may not work with other URLs, defaults to 'https://mof.tech.northwestern.edu/mofs/#.json'
    :type url: str, optional
    :param headers: dictionary of search parameters, defaults to {'loading': 'cm3(STP)/cm3', 'pressure': 'bar'}
    :type headers: dict, optional
    :param params: dictionary of the headers to be output in the result, defaults to {'vf_min': 0.75, 'vf_max': 1, 'database':'COREMOF 2019','page':1}
    :type params: dict, optional
    :param npages: number of the result pages to be output, defaults to 1
    :type npages: int, optional
    :param limit: semaphore limit of number of jobs per block, no need to change this, defaults to 1000
    :type limit: int, optional
    :return: an instance of asyncio.Future that contains all the search results
    :rtype: asyncio.Future
    """
    from furl import furl
    import asyncio

    tasks = []
    from rich.progress import track

    # create instance of Semaphore
    sem = asyncio.Semaphore(limit)
    for i in track(range(1, npages + 1)):
        url_string = furl(url)
        params["page"] = i
        url_string.set(params)

        # pass Semaphore and session to every GET request
        task = asyncio.ensure_future(bound_fetch(sem, url_string.tostr(), session))
        tasks.append(task)

    responses = asyncio.gather(*tasks)
    return await responses

def get_data_from_mofdb(
    url="https://mof.tech.northwestern.edu/mofs/#.json",
    params={"vf_min": 0.1, "vf_max": 1.0, "database": "IZA", "page": 1},
    headers={"loading": "cm3(STP)/cm3", "pressure": "bar"},
    npages=None,
    limit=1000,
):
    """
    This is a wrapper for the MOFDB API, which is explained here:https://mof.tech.northwestern.edu/api
    It takes in a url, parameters, headers, number of pages and a path to the mofography repository and
    returns a pandas dataframe of the data.

    The function is asynchronous and uses the aiohttp library to make multiple requests to the MOFDB
    API.



    :param url: The url of the MOF database, defaults to https://mof.tech.northwestern.edu/mofs/#.json
    (optional)
    :param params: a dictionary of parameters to be passed to the url. The default parameters are: {'vf_min': 0.1, 'vf_max': 1.0, 'database':'IZA','page':1}
    :param headers: This is the headers that you want to pass to the MOFDB API. The default is to get
    the loading and pressure data
    :param npages: number of pages to be scraped. If None, then the number of pages is determined by the
    function to return all the pages
    :param limit: maximum number of pages allowed in the query (no need to change this one unless overflow )
    :return: A dataframe with the results of the query.
    """

    from aiohttp import ClientSession
    import asyncio
    import numpy as np
    import nest_asyncio

    nest_asyncio.apply()
    import pandas as pd

    # just put page:1 in the params it doesn't matter, only the npages parameter is used
    params["page"] = 1

    if npages == None:
        npages = get_number_of_pages(url, headers, params)

    # This shouldn't be called in a function (similar to initiating a Dask cluster)
    loop = asyncio.get_event_loop()
    # with ClientSession(headers= headers) as session:
    session = ClientSession(headers=headers)
    fut = asyncio.ensure_future(
        get_mofdb_data(
            session, url=url, params=params, headers=headers, npages=npages, limit=limit
        )
    )
    loop.run_until_complete(fut)
    session.close()
    data_df = pd.DataFrame(list(np.hstack(([f["results"] for f in fut.result()]))))

    return data_df

def get_number_of_pages(url, headers, params):
    """
    Get the number of pages of the result, from the url, for a given set of headers and parameters

    :param url: url for MOFDb for example
    :type url: string
    :param headers: dictionary of headers
    :type headers: dict
    :param params: dictionary of parameters
    :type params: dict
    :return: number of pages in the result
    :rtype: int
    """

    import requests

    resp = requests.get(url=url, headers=headers, params=params)
    resp = resp.json()
    return resp["pages"]