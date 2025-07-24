"""
Module Name:    utrpy_gff_utils.py
Description:    Functions for pandas.Dataframe represented GFF-files (alphabetically sorted)
Author:         Simon Hegele
Date:           2025-04-01
Version:        1.1
License:        GPL-3
"""

from pandas import DataFrame, Series, read_csv
from typing import Callable, Generator

gff_columns = ["seqname",
               "source",
               "type",
               "start",
               "end",
               "score",
               "strand",
               "frame",
               "attributes"]

class NoAncestorFoundException(Exception):
    
    def __init__(self, type: str, feature_id: str, trace: list[Series]):

        trace_IDs = [attributes_dict(feature)["ID"] for feature in trace]
        trace_str = "\n".join(id for id in trace_IDs)
        error_msg = f"No {type} ancestor for {feature_id},\nTrace:\n{trace_str}"

        super().__init__(error_msg)

class MultipleAncestorsFoundException(Exception):
    
    def __init__(self, type: str, feature_id: str, trace: list[str], ancestors: DataFrame):

        trace_IDs = [attributes_dict(feature)["ID"] for feature in trace]
        ances_IDs = [attributes_dict(a)["ID"] for _, a in ancestors.iterrows()]
        trace_str = "\n".join(id for id in trace_IDs)
        error_msg  = f"Multiple {type} ancestors ({ances_IDs}) for {feature_id},\nTrace:\n{trace_str}"

        super().__init__(error_msg)

def attributes_dict(feature: Series) -> dict[str, str]:
    """
    Parsing the key=value pairs from a features attributes fields into a hashmap
    """
    
    return {a.split("=")[0]: a.split("=")[1] for a in feature["attributes"].split(";")}

def attributes_str(attributes: dict[str, str]) -> str:
    
    return ";".join([f"{key}={attributes[key]}" for key in attributes.keys()])

def check_strands(feature1: Series,
                  feature2: Series,
                  know_strand=False) -> bool:

    if know_strand:
        if not feature1["strand"]=="." or feature2["strand"]==".":
            if feature1["strand"]==feature2["strand"]:
                return True
    else:
        if feature1["strand"]=="." or feature2["strand"]==".":
            return True
        if feature1["strand"]==feature2["strand"]:
            return True
    return False

def empty_gff() -> DataFrame:
    """
    Creates and returns an empty GFF-file
    """
    return DataFrame({}, columns = gff_columns)

def get_ancestor(gff: DataFrame,
                 feature: Series,
                 type: str) -> Series:
    """
    Returns the ancestor of the specified type for the input feature
    a) Directly, if <ancestor_type>_id=ancestor is in the attributes column or
    b) By recursively following the GFF-hierarchy following the child-parent relationship
    """

    feature_id = attributes_dict(feature)["ID"]
    ancestor   = feature
    trace      = []

    for _ in range(5):

        trace.append(ancestor)

        if type==(ancestor["type"]):
            return ancestor
        
        # Retrieving ID of Parent / Ancestor
        attrs = attributes_dict(ancestor)
        if f"{type}_id" in attrs.keys():
            ancestor_id = attrs[f"{type}_id"]
        elif type in attrs.keys():
            ancestor_id = attrs[type]
        elif "Parent" in attrs.keys():
            ancestor_id = attrs["Parent"]
        else:
            raise NoAncestorFoundException(type, feature_id, trace)
        
        # Retrieving Parent / Ancestor
        ancestors = gff[gff.apply(lambda f: attributes_dict(f)["ID"]==ancestor_id, axis=1)]
        if len(ancestors) == 0:
            raise NoAncestorFoundException(type, feature_id, trace)
        if len(ancestors) > 1:
            raise MultipleAncestorsFoundException(type, feature_id, trace, ancestors)
        ancestor = ancestors.iloc[0]

def get_subtree(gff: DataFrame, feature: Series) -> DataFrame:

    prefiltered   = overlapping_features(gff, feature)
    descends_mask = prefiltered.apply(lambda f: is_descendant(prefiltered, f, feature), axis=1)

    return prefiltered[descends_mask]

def feature_includes(feature_1: Series,
                     feature_2: Series) -> bool:
    """
    Checks if the genomic location of feature_1 includes the genomic location of feature_2
    """

    if not feature_1["seqname"] == feature_2["seqname"]:
        return False
    if not feature_1["start"] <= feature_2["start"]:
        return False
    if not feature_1["end"] >= feature_2["end"]:
        return False
    return True

def features_overlap(feature_1: Series,
                     feature_2: Series) -> bool:
    """
    Checks if two features overlap
    """

    if feature_1["seqname"] == feature_2["seqname"]:
        if (feature_1["start"]<=feature_2["end"]) and (feature_1["end"]>=feature_2["start"]):
            return True
        if (feature_2["start"]<=feature_1["end"]) and (feature_2["end"]>=feature_1["start"]):
            return True
    return False

def feature_pairs(gff_1: DataFrame,
                  gff_2: DataFrame,
                  pairing: Callable[[DataFrame, Series],DataFrame],
                  type="") -> Generator:
    
    gff_1_features = gff_1.loc[gff_1["type"].str.contains(type, regex=False)]
    gff_2_features = gff_2.loc[gff_2["type"].str.contains(type, regex=False)]

    for i, gff_1_feature in gff_1_features.iterrows():

        for j, gff_2_feature in pairing(gff_2_features, gff_1_feature).iterrows():

            yield gff_1_feature, gff_2_feature
    
def load_gff(file_path: str) -> DataFrame:
    """
    Loading a GFF-file from the file-system
    """
    return read_csv(file_path, sep="\t", header=None, comment="#", names=gff_columns)

def included_features(gff: DataFrame,
                      feature: Series,
                      type="") -> DataFrame:
    """
    Returns features of the specified type from the input GFF included by the input feature 
    """
    mask_seqname  = gff["seqname"] == feature["seqname"]
    mask_type     = gff["type"].str.contains(type, regex=False)
    prefiltered   = gff[mask_seqname & mask_type]
    mask_includes = prefiltered.apply(lambda f: feature_includes(feature, f), axis=1)

    return prefiltered[mask_includes]

def including_features(gff: DataFrame,
                      feature: Series,
                      type="") -> DataFrame:
    """
    Returns features of the specified type from the input GFF included by the input feature 
    """
    mask_seqname  = gff["seqname"] == feature["seqname"]
    mask_type     = gff["type"].str.contains(type, regex=False)
    prefiltered   = gff[mask_seqname & mask_type]
    mask_includes = prefiltered.apply(lambda f: feature_includes(f, feature), axis=1)

    return prefiltered[mask_includes]

def is_descendant(gff: DataFrame,
                  feature_1: Series,
                  feature_2: Series) -> bool:
    
    try:
        return feature_2.equals(get_ancestor(gff, feature_1, feature_2["type"]))
    except NoAncestorFoundException:
        return False    

def overlapping_features(gff: DataFrame,
                         feature: Series,
                         type="") -> DataFrame:
    """
    Returns features of the specified type from the input GFF overlapping the input feature 
    """
    
    mask_seqname = gff["seqname"] == feature["seqname"]
    mask_type    = gff["type"].str.contains(type, regex=False)
    prefiltered  = gff[mask_seqname & mask_type]
    mask_overlap = prefiltered.apply(lambda f: features_overlap(feature, f), axis=1)

    return prefiltered[mask_overlap]

def seqname_split(gff: DataFrame,
                  seqnames=None) -> dict[str, DataFrame]:

    if seqnames is None:
        seqnames = gff["seqname"].unique()

    return {seqname: gff.loc[gff["seqname"]==seqname]
            .sort_values("start")
            .reset_index(drop=True)
            for seqname in seqnames}

def to_string(feature: Series):

    return "\t".join(str(v) for v in list(feature))

def type_split(gff: DataFrame, type: str) -> Generator[DataFrame, None, None]:
    
    for i, feature in gff.loc[gff["type"].str.contains(type, regex=False)].iterrows():

        yield get_subtree(gff, feature)

def write_gff(gff: DataFrame, file_path: str, mode="w") -> None:

    gff.to_csv(file_path, sep="\t", index=False, header=None, mode=mode)