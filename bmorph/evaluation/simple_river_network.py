import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from bmorph.evaluation.plotting import find_upstream
from bmorph.evaluation import plotting
import networkx as nx

class SegNode():
    """
    SegNode
        creates a node of a segment to be used in the simple
        river network.
    ---
    attributes
        pfaf_code:
            the pfafstetter code for this seg
        seg_id:
            the id for this seg
        upstream:
            a List[SegNode] containing what is
            directly upstream from this SegNode
        basin_area:
            the summative Basin Area for this seg
        end_marker:
            a boolean noting if this node marks
            the end of a basin, primarily used
            in simple_river_network.ecode_pfaf
        encoded:
            a boolean noting if this node has
            been fully given a unique pfaf_code
    """
    def __init__(self, seg_id, pfaf_code):
        self.pfaf_code = pfaf_code
        self.seg_id = seg_id
        self.aggregated_seg_ids = list()
        self.upstream: List[SegNode] = list()
        self.basin_area: None

        self.end_marker = False
        self.encoded = False

    def __repr__(self):
         return f'[seg_id: {self.seg_id}, pfaf_code: {self.pfaf_code}]'

    def __str__(self):
        upstream_seg_id = list()
        for upstream_SegNode in self.upstream:
            upstream_seg_id.append(upstream_SegNode.seg_id)
        return f'seg_id: {self.seg_id}, pfaf_code: {self.pfaf_code}, upstream: {upstream_seg_id}'

    def __eq__(self, other):
        if isinstance(other, SegNode):
            return ((self.seg_id == other.seg_id)
                    and (self.pfaf_code==other.pfaf_code)
                    and (self.basin_area==other.basin_area))

    def __iter__(self):
        yield self
        for upstream in self.upstream:
            for node in upstream:
                yield node


class SimpleRiverNetwork:
    """
    SimpleRiverNetwork
        the simple river network maps nodes within a
        given topography to visualize their arragments
        and simplify different parts of the network
        to track statistics propogating through the network
    ----
    attributes:
        topo:
            the xrray Dataset of the topography.
        seg_id_values:
            a list of all seg_id's being used in the network.
        outlet:
            the end of the river network and start of the
            simple river network, aka "root.""
        adj_mat:
            a numpy adjacency matrix that can be used to graph
            the simple river network.
    functions:
        parse_upstream:
            recursively constructs network by searching what
            SegNodes are upstream of the current SegNode
            and updates the current SegNode's upstream list
            while also building the adjacency matrix.
        collect_upstream_nodes
            finds all nodes upstream of a node and returns
            a list of them
        clear_network:
            sets the adjacency matrix to an empty array and
            sets the upstream designation of the outlet to an
            empty list, clearing the shape of the network
            upstream of the outlet. This does not reset
            the topograpghy or seg_id_values, so the original
            shape can be rebuilt.
        clear_end_markers
            sets all end_mark in nodes at and upstream of node
            to False
        update_node_area
            updates the desired node with basin area information
        net_upstream_area
            calculates the basin area upstream of node of interest.
            This does include the area of the node of interest
        force_upstream_area
            operates the same as net_upstream_area, but
            ignores end_marking
        check_upstream_end_marking
            checks if any nodes directly upstream are
            end_markers and returns True if so
        count_net_upstream
            counts the number of nodes upstream of a
            node, including the original node
        find_branch
            locates a node that branches into 2+ nodes,
            returning what node branches and any
            nodes prior to the branch taht where in a row
        find_node
            searches for and returns a node with the desired
            seg_id upstream of a starting_node. If the node
            cannot be found, None is returned
        find_like_pfaf
            finds all nodes with the matching digit at the exact
            same location in pfaf_code and returns all of them
            in a list
        append_pfaf
            adds a pfaffstetter code digit to all upstream nodes
        append_sequential
            adds odd digits to the pfaf_codes of SegNodes in
            a row, or in sequence. this ensures all SegNodes
            within the SimpleRiverNetwork have a unique code
        sort_streams
            returns which branches are part of the mainstream and which
            are part of the tributaries
        find_tributary_basins
            finds the four tributaries with the largest drainage areas
        encode_pfaf
            recursively encodes pfafstetter codes on a SimpleRiverNetwork
        sort_by_pfaf
            sorts a list of SegNode's in decreasing order
            of a pfaf_digit at the given degree
        generate_pfaf_map
            creates a list of pfaf_code values in the order
            of the seg_id_values, including the seg_id_values.
            this is a little more cluttered, reccommended only
            for debugging purposes
        generate_pfaf_codes
            creates a list of pfaf_code values in the order
            of the seg_id_values
        generate_weight_map
            creates a list of fractional weights equivalent
            to the node's upstream area divided by the overall
            basin_area of the whole SimpleRiverNetwork.
            these are in order of the seg_id_values
        pfaf_aggregate
            aggregates the flow network by one pfafstetter level
        draw_network
                plots the network through networkx
        draw_multi_measure
                plots several networkx plots of user specified transparency for a single
                SimpleRiverNetwork to compare mutliple measures at once
        generate_mainstream_map
            creates a list of which nodes are part of the
            mainstream in order of the seg_id_values
        generate_pfaf_color_map
            creates a pd.Series to assign a unqiue color to each
            first level pfafstetter basin
        generate_node_highlight_map
            takes a list of seg_ids and creats a pd.Series
            that highlights the nodes in the list
        reconstruct_adj_mat
            rebuilds the adjacency matrix from an existing flow tree
        pfaf_aggregate
            aggregates the flow network by one pfafstetter level
        aggregate_measure_sum
            determines the sum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        aggregate_measure_mean
            determines the mean measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        aggregate_measure_median
            determines the median measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        aggregate_measure_max
            determines the maximum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        aggregate_measure_min
            determines the minimum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        aggregate_measure
            aggregates the measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        spawn_srn
            creates a new SimpleRiverNetwork from that
            given network and upstream of it
    """
    def __init__(self, topo: xr.Dataset, pfaf_seed = int, outlet_index = 0, max_pfaf_level=42):
        self.topo = topo
        self.seg_id_values = topo['seg_id'].values
        self.outlet = SegNode(seg_id=self.seg_id_values[outlet_index], pfaf_code=str(pfaf_seed))

        self.update_node_area(self.outlet)

        N = topo.dims['seg']
        self.adj_mat = np.zeros(shape=(N, N), dtype=int)

        self.parse_upstream(self.outlet)
        self.encode_pfaf(self.outlet,max_level=max_pfaf_level)
        self.network_graph = plotting.create_nxgraph(self.adj_mat)
        self.network_positions = plotting.organize_nxgraph(self.network_graph)

        self.clear_end_markers(self.outlet)


    def __eq__(self, other):
        if isinstance(other, SimpleRiverNetwork):
            return self.branch_eq_(self.outlet,other.outlet)


    def branch_eq_(self, node_self, node_other, match=True):
        """
        branch_eq_
            used in the __eq__ function for SimpleRiverNetwork as
            a recursive function to see if two node's upstream
            structure are equivalent
        ----
        node_self:
            a SegNode to compare with node_other
        node_other:
            a SegNode to compare with node_self
        match:
            holds the default assumption on whether
            or not the two upstream structures are equivalent,
            (warning: this function will only change the assumption
            to False, changning this may result in a false negative)
        return:
            whether or not the upstream structures are equivalent
        """
        # we are settng the default agrument to True and
        # will change it only to false throughout the code
        # so we know there are no False statements that
        # are overriden (i.e. no False Falses)
        if node_self == node_other and match:
            if node_self.upstream and node_other.upstream:
                for upstream_self,upstream_other in zip(node_self.upstream,node_other.upstream):
                    match = self.branch_eq_(upstream_self,upstream_other,match)
            elif node_self.upstream or node_other.upstream:
                # meaning one continues upstream while the other
                # doesn't and is therefor inequivalent
                match = False
        else:
            # meaning the nodes don't match
            # or a mismatch was already found
            match = False

        return match

    def parse_upstream(self, node: SegNode):
        """
        parse_upstream:
            recursively constructs network by searching what
            SegNodes are upstream of the current SegNode
            and updates the current SegNode's upstream list
            while also building the adjacency matrix.
        ----
        node:
            a SegNode to start building the network from
        """
        upstream_seg_indices = list()
        node_seg_index = np.where(self.seg_id_values == node.seg_id)
        find_upstream(self.topo, node.seg_id,upstream_seg_indices)

        for upstream_seg_index in upstream_seg_indices:
            self.adj_mat[node_seg_index, upstream_seg_index] += 1
            upstream_seg_id = self.seg_id_values[upstream_seg_index]
            upstream_node = SegNode(seg_id=upstream_seg_id, pfaf_code=self.outlet.pfaf_code)
            self.update_node_area(upstream_node)
            node.upstream.append(upstream_node)
            self.parse_upstream(upstream_node)


    def collect_upstream_nodes(self, node:SegNode):
        """
        collect_upstream_nodes
            finds all nodes upstream of a node and returns
            a list of them
        ----
        node:
            a SegNode to collect upstream nodes from
        return:
            a list of all upstream nodes
        """
        upstream_nodes = list()
        upstream_nodes.extend(node.upstream)
        for upstream in node.upstream:
            upstream_nodes.extend(self.collect_upstream_nodes(upstream))
        return upstream_nodes

    def clear_network(self):
        """
        clear_network:
            sets the adjacency matrix to an empty array and
            sets the upstream designation of the outlet to an
            empty list, clearing the shape of the network
            upstream of the outlet. This does not reset
            the topograpghy or seg_id_values, so the original
            shape can be rebuilt.
        """
        self.adj_mat = np.zeros(shape = (0))
        self.outlet.upstream = list()

    def clear_end_markers(self,node):
        """
        clear_end_markers
            sets all end_mark in nodes at and upstream of node
            to False
        """
        if node:
            node.end_marker = False
            for upstream in node.upstream:
                self.clear_end_markers(upstream)

    def clear_end_markers(self,node):
        """
        clear_end_markers
            sets all end_mark in nodes at and upstream of node
            to False
        """
        if node:
            node.end_marker = False
            for upstream in node.upstream:
                self.clear_end_markers(upstream)

    def update_node_area(self, node: SegNode):
        """
        update_node_area
            updates the desired node with basin area information
        ----
        node:
            a SegNode to change it and only its basin_area
        """
        basin_area_indices = np.where(self.topo['seg_hru_id'] == node.seg_id)[0]
        basin_area = 0
        for basin_area_index in basin_area_indices:
            basin_area += self.topo['Basin_Area'].values[basin_area_index]
        node.basin_area = basin_area

    def net_upstream_area(self, node: SegNode):
        """
        net_upstream_area
            calculates the basin area upstream of node of interest.
            This does include the area of the node of interest
        ----
        node:
            a SegNode to start from and calculate both its and
            all upstream nodes aggregate area
        return:
            returns the aggregate upstream_area
        """
        net_area = 0
        if node and not node.end_marker:
            net_area = node.basin_area

            for upstream_node in node.upstream:
                try:
                    net_area += self.net_upstream_area(upstream_node)
                except RecursionError as e:
                    print(node, '\n', node.upstream,"\n", upstream_node,"\n", net_area)
                    raise RecursionError('no')

        return net_area

    def force_upstream_area(self, node:SegNode):
        """
        force_upstream_area
            operates the same as net_upstream_area, but
            ignores end_marking
        """
        net_area = 0
        net_area = node.basin_area

        for upstream_node in node.upstream:
            net_area += self.force_upstream_area(upstream_node)

        return net_area

    def check_upstream_end_marking(self, node: SegNode):
        """
        check_upstream_end_marking
            checks if any nodes directly upstream are
            end_markers and returns True if so
        ----
        node:
            a SegNode to check directly upstream from
        """
        end_marker_ahead = False
        for upstream_node in node.upstream:
            if upstream_node.end_marker:
                end_marker_ahead = True

        return end_marker_ahead

    def count_net_upstream(self,node:SegNode):
        """
        count_net_upstream
            counts the number of nodes upstream of a
            node, including the original node
        ----
        node:
            a SegNode to begin counting from
        return:
            a count of the number of nodes upstream
        """
        count = 0
        if node and not node.end_marker:
            count += 1
            for upstream_node in node.upstream:
                count += self.count_net_upstream(upstream_node)
        return count

    def find_branch(self, node:SegNode):
        """
        find_branch
            locates a node that branches into 2+ nodes,
            returning what node branches and any
            nodes prior to the branch taht where in a row
        ----
        node:
            a SegNode to start searching from
        return:
            branch, the SegNode that is branching
            sequential_nodes, a list of nodes in a row
                prior to and including branch
        """
        branch = None
        orig_node = node
        sequential_nodes = []
        while not node.end_marker and node.upstream and not branch:
            if len(node.upstream) > 1:
                branch = node
            else:
                node = node.upstream[0]
                sequential_nodes.append(node)
        if len(sequential_nodes): sequential_nodes = [orig_node] + sequential_nodes
        return branch, sequential_nodes

    def find_node(self, target_id, node:SegNode):
        """
        find_node
            searches for and returns a node with the desired
            seg_id upstream of a starting_node. If the node
            cannot be found, None is returned
        ----
        target_id:
            a seg_id to search for within the SimpleRiverNetwork
        node:
            a SegNode to start searching from
        """
        if node:
            if node.seg_id == target_id:
                return node
            else:
                if node.upstream:
                    i = 0
                    result = None
                    while not result and i < len(node.upstream):
                        result = self.find_node(target_id, node.upstream[i])
                        i += 1
                    return result
                else:
                    return None

    def find_like_pfaf(self, node:SegNode, target_pfaf_digits: list, degree:int):
        """
        find_like_pfaf
            finds all nodes with the matching digit at the exact
            same location in pfaf_code and returns all of them
            in a list
        ----
        node:
            a SegNode to start searching from
        target_pfaf_digits:
            a list of pfaf_digit to search for in the SimpleRiverNetwork
        degree:
            how many pfafstetter levels deep should the function look for
            i.e. if you have degree 2, and a pfaf_code of 1234, it will
            look at "3" to see if is a match
        return:
            a list of like_pfaf_nodes that match the target_pfaf_digits
            at the input degree
        """
        like_pfaf_nodes= list()
        if node:
            if degree < len(node.pfaf_code) and (int(node.pfaf_code[degree]) in target_pfaf_digits):
                like_pfaf_nodes.append(node)
            for upstream in node.upstream:
                like_pfaf_nodes.extend(self.find_like_pfaf(upstream, target_pfaf_digits, degree))
        return like_pfaf_nodes

    def append_pfaf(self, node: SegNode, pfaf_digit:int):
        """
        append_pfaf
            adds a pfaffstetter code digit to all upstream nodes
        ----
        node: a SegNode to designate the root of the flow tree

        pfaf_digit: the digit to be added to the pfaffstetter codes
        """
        if not node.end_marker:
            node.pfaf_code += str(pfaf_digit)
            for upstream_node in node.upstream:
                self.append_pfaf(upstream_node, pfaf_digit)

    def append_sequential(self, sequence, base=''):
        """
        append_sequential
            adds odd digits to the pfaf_codes of SegNodes in
            a row, or in sequence. this ensures all SegNodes
            within the SimpleRiverNetwork have a unique code
        ----
            sequence:
                a list of SegNodes in a sequence, typically
                aggregated from find_branch
            base:
                an addition to the pfaf_code
        """
        if not len(sequence): return
        #log5 is used since there are 5 odd digits in range(1,10)
        #and we want to append purely odd digits for sequential nodes
        total_digits = int(np.ceil(np.max([np.log(len(sequence))/np.log(5), 1])))
        append_digit = int(total_digits*'1')
        for seq_node in sequence:
            if not seq_node.encoded:
                seq_node.pfaf_code += str(append_digit) + base
                seq_node.encoded = True
            append_digit += 2

    def sort_streams(self, node = SegNode):
        """
        sort_streams
            returns which branches are part of the mainstream and which
            are part of the tributaries
        ----
        node: a SegNode to start tracing the mainstream from.
            this is the "root" of the flow tree
        return:
            mainstreams:
                a list of mainstream SegNodes determined by having
                the greatest upstream_area
            tributaries:
                a list of tributaries having upstream_area less than
                the mainstream but still encountered along the way.
        """
        tributaries = list()
        mainstreams = list()

        mainstreams.append(node)

        while node.upstream and not self.check_upstream_end_marking(node):

            if len(node.upstream) == 1:
                #if there is only one upstream node, i.e. no branching
                node = node.upstream[0]
                mainstreams.append(node)
            else:
                #find which node has the largest upstream area and assign
                #it as the mainstreams and the rest as tributaries
                upstream_areas = list()

                for upstream_node in node.upstream:
                    upstream_areas.append(self.net_upstream_area(upstream_node))

                max_upstream_area = np.max(upstream_areas)
                max_upstream_index = upstream_areas.index(max_upstream_area)
                mainstream_node = node.upstream[max_upstream_index]
                mainstreams.append(mainstream_node)

                for tributary_node in node.upstream:
                    if tributary_node is not mainstream_node:
                        #Non-binary definitions of streams coming soon ....
                        tributaries.append(tributary_node)

                node = mainstream_node

        return mainstreams, tributaries

    def find_tributary_basins(self, tributaries):
        """
        find_tributary_basins
            finds the four tributaries with the largest drainage areas
        ----
        tributaries:
            a list of tributary SegNodes to be searched
        return:
            returns a list of the largest_tributaries found
            in the list of tributaries given.
        """
        largest_tributaries = list()
        largest_tributary_areas = list()

        if len(tributaries) > 4:
            for tributary in tributaries:
                if len(largest_tributaries) < 4:
                    #keep adding until there are 4 values
                    largest_tributaries.append(tributary)
                    largest_tributary_areas.append(self.net_upstream_area(tributary))
                else:
                    #then begin prioritizing by upstream_area
                    #aparently the target audience was not as excited by the smaller ones
                    tributary_upstream_area = self.net_upstream_area(tributary)
                    if tributary_upstream_area > np.min(largest_tributary_areas):
                        #the minimum will be the value needing to be replaced and is
                        #therefore the tested case so that we end up with a list of
                        #nothing being greater than the smallest value
                        remove_index = np.where(largest_tributary_areas
                                            == np.min(largest_tributary_areas))[0][0]
                        del largest_tributaries[remove_index]
                        del largest_tributary_areas[remove_index]
                        largest_tributaries.append(tributary)
                        largest_tributary_areas.append(tributary_upstream_area)
            return largest_tributaries
        else:
            #either:
            #a) there were no tributaries to begin with
            #b) you had excatly the max number (4)
            #c) there were less than 4 and you dont need more
            return tributaries

    def encode_pfaf(self, root_node = SegNode, level=0, max_level=42):
        """
        encode_pfaf
            recursively encodes pfafstetter codes on a SimpleRiverNetwork
        ----
        root_node:
            a SegNode from which to start encoding
        level:
            how many levels deep into recursion the method already is
        max_level:
            the maximum number of levels encode_pfaf will run for
            before raising a RuntimeError
        """
        if level > max_level:
            raise RuntimeError(f"encode_pfaf has exceed the maximum level of recurion: {max_level}"
                               f"\n You may specify the max_level as a keyword argument if you require more digits")

        mainstreams, tributaries = self.sort_streams(root_node)

        #find tributaries with largest draingage areas
        tributary_basins = self.find_tributary_basins(tributaries)
        #tributary_basins is in upstream order since tributaries
        #was made in upstream order

        pfaf_digit=1
        if tributary_basins:

            tributary_basin_index = 0
            interbasin_root = root_node
            all_seq_nodes = []

            for mainstream_index,mainstream in enumerate(mainstreams):

                while (tributary_basin_index < len(tributary_basins)
                        and tributary_basins[tributary_basin_index] in mainstream.upstream):

                    tributary_basin_found = tributary_basins[tributary_basin_index]

                    #interbasin handling:
                    #1. mark off the interbasin from the rest of the flow tree
                    #2. assign pfaf codes to interbasin members from interbasin_root
                    #3. recursion call the interbasin for pfaf_encode
                    #4. unmark upstream and continue parsing
                    #5. update interbasin_root
                    for upstream_node in mainstream.upstream:
                        upstream_node.end_marker = True

                    self.append_pfaf(interbasin_root, pfaf_digit)
                    self.encode_pfaf(interbasin_root, level=level+1)
                    branch, snodes = self.find_branch(interbasin_root)
                    if len(snodes):
                        all_seq_nodes.append(snodes)

                    for upstream_node in mainstream.upstream:
                        upstream_node.end_marker = False

                    interbasin_root = mainstreams[mainstream_index+1]
                    pfaf_digit += 1 #converts to even to prep for tributary_basin

                    #tributary_basin handling
                    self.append_pfaf(tributary_basin_found, pfaf_digit)
                    self.encode_pfaf(tributary_basin_found, level=level+1)
                    branch, snodes = self.find_branch(tributary_basin_found)
                    if len(snodes):
                        all_seq_nodes.append(snodes)
                    tributary_basin_index += 1
                    pfaf_digit += 1 #then updates to odd to prep for interbasin

                #this is for the final interbasin basin of the pfaf_code
                #where we have passed all of our tributary basins
                #we also check if this is simply part of a smaller flow tree
                #by checking end_marking
                is_part_of_flow_tree = (tributary_basin_index == len(tributary_basins)
                                        and self.check_upstream_end_marking(mainstream))
                if not mainstream.upstream or is_part_of_flow_tree:
                    self.append_pfaf(interbasin_root, pfaf_digit)
                    self.encode_pfaf(interbasin_root, level=level+1)
                    branch, snodes = self.find_branch(interbasin_root)
                    if len(snodes):
                        all_seq_nodes.append(snodes)

            snodes = list(all_seq_nodes)

            for sn in snodes:
                self.append_sequential(sn, base='')

    def sort_by_pfaf(self,nodes:list,degree=int):
        """
        sort_by_pfaf
            sorts a list of SegNode's in decreasing order
            of a pfaf_digit at the given degree
        ----
        nodes:
            a list of SegNodes to sort
        degree:
            which index in a pfaf_code the nodes
            are to be sorted by
        return:
            returns the list of sorted nodes
        """
        sorted_nodes = list()
        digit = 9
        while len(nodes) > 0 and digit > -1:
            index = 0
            for node in nodes:
                if int(node.pfaf_code[degree])==digit:
                    sorted_nodes.append(nodes[index])
                index += 1
            digit -= 1
        return sorted_nodes

    def generate_pfaf_map(self):
        """
        generate_pfaf_map
            creates a list of pfaf_code values in the order
            of the seg_id_values, including the seg_id_values.
            this is a little more cluttered, reccommended only
            for debugging purposes
        """
        pfaf_map = list()
        for i, seg_id in enumerate(self.seg_id_values):
            node = self.find_node(seg_id, self.outlet)
            if node:
                pfaf_map.append(f'{int(node.seg_id)}-{node.pfaf_code}')
        return pfaf_map

    def generate_pfaf_codes(self):
        """
        generate_pfaf_codes
            creates a list of pfaf_code values in the order
            of the seg_id_values
        """
        pfaf_map = list()
        for i, seg_id in enumerate(self.seg_id_values):
            node = self.find_node(seg_id, self.outlet)
            pfaf_map.append(int(node.pfaf_code))
        return pfaf_map

    def generate_weight_map(self):
        """
        generate_weight_map
            creates a list of fractional weights equivalent
            to the node's upstream area divided by the overall
            basin_area of the whole SimpleRiverNetwork.
            these are in order of the seg_id_values
        """
        weight_map = list()
        total_area = self.topo['Basin_Area'].values.sum()
        for i, seg_id in enumerate(self.seg_id_values):
            node = self.find_node(seg_id, self.outlet)
            area = self.net_upstream_area(node)
            area_fraction = area/total_area
            weight_map.append(f'{i}-{area_fraction}')
        return weight_map

    def generate_mainstream_map(self):
        """
        generate_mainstream_map
            creates a list of which nodes are part of the
            mainstream in order of the seg_id_values
        """
        mainstream, tributaries = self.sort_streams(self.outlet)
        mainstream_ids = list()
        for mainstream_node in mainstream:
            mainstream_ids.append(mainstream_node.seg_id)
        all_segs = list(map(int,self.seg_id_values))
        mainstream_seg_map = pd.Series(all_segs).isin(mainstream_ids).astype(int)
        return mainstream_seg_map

    def generate_pfaf_color_map(self):
        """
        generate_pfaf_color_map
            creates a pd.Series to assign a unqiue color to each
            first level pfafstetter basin
        """
        pfaf_color_map = list()
        for i,seg_id in enumerate(self.seg_id_values):
            node = self.find_node(seg_id, self.outlet)
            first_pfaf_digit = int(node.pfaf_code[0])
            pfaf_color_map.append(first_pfaf_digit)
        return pd.Series(pfaf_color_map)

    def generate_node_highlight_map(self,seg_ids:list):
        """
        generate_node_highlight_map
            takes a list of seg_ids and creats a pd.Series
            that highlights the nodes in the list
        ----
        seg_ids:
            a list of seg_id values that are to be marked
            apart from the rest of the seg_ids
        return:
            a list that will identify these highlighted
            nodes for draw_network
        """
        return pd.Series(self.seg_id_values).isin(seg_ids).astype(int)

    def reconstruct_adj_mat(self, node, adj_mat: np.ndarray):
        """
        reconstruct_adj_mat
            rebuilds the adjacency matrix from an existing flow tree
        ----
        node:
            a node to construct the adjacency matrix from
        adj_mat:
            a square numpy ndarray of zeros originally, the size equal to
            len(seg_id_values) by len(seg_id_values), to be filled
        return:
            returns the reconstructed adjacenecy matrix that needs
            to be set as the network's adjacency matrix to actually
            alter the flow tree
        """
        if node and node.upstream:
            node_seg_index = np.where(self.seg_id_values == node.seg_id)

            for upstream_node in node.upstream:
                upstream_seg_index = np.where(self.seg_id_values == upstream_node.seg_id)
                adj_mat[node_seg_index, upstream_seg_index] += 1
                adj_mat = self.reconstruct_adj_mat(upstream_node,adj_mat)

        return adj_mat

    def pfaf_aggregate(self):
        """
        pfaf_aggregate
            aggregates the flow network by one pfafstetter level
        """
        # we will be looking to reduce the total number of levels by 1
        # in each aggregation cycle
        # this means that we will look to aggregate the deepest level
        # and search the flow tree for pfaf_codes matching the length
        # of the maximum level

        pfaf_codes = self.generate_pfaf_codes()
        current_total_levels = max([len(s.pfaf_code) for s in self.outlet])

        if current_total_levels == 1:
            raise UserError("The network currently has a maximum Pfaffstetter code of length 1 - cannot aggregate any further!")

        aggregation_target_nodes = self.find_like_pfaf(self.outlet, list(np.arange(10)), current_total_levels-1)
        aggregation_target_nodes = self.sort_by_pfaf(aggregation_target_nodes,current_total_levels-2)

        search_index = 0
        if current_total_levels > -1:
            while search_index < len(aggregation_target_nodes):

                current_node = aggregation_target_nodes[search_index]

                # we need to collect which nodes we are going to aggregate
                summative_basin_area = 0
                basin_nodes = list()
                aggregated_seg_ids = list()
                basin_pfaf_root = current_node.pfaf_code[:-1]

                for target_node in aggregation_target_nodes:
                    if target_node.pfaf_code[:-1] == basin_pfaf_root and target_node != current_node:
                        basin_nodes.append(target_node)
                        summative_basin_area += target_node.basin_area

                        aggregated_seg_ids.append(target_node.seg_id)
                        aggregated_seg_ids.extend(target_node.aggregated_seg_ids)

                # since we are going to aggregate these upsteam nodes
                # into the current node, we need to remove them
                # from our search list to avoid searching for ghost nodes
                # also, be sure to check out my new techno-thriller: ghost_nodes
                for basin_node in basin_nodes:
                    aggregation_target_nodes.remove(basin_node)
                    self.seg_id_values = np.delete(self.seg_id_values,np.where(
                        self.seg_id_values == basin_node.seg_id))

                #print(" basin_nodes: ", basin_nodes)

                basin_upstream = list()
                for basin_node in basin_nodes:
                    if basin_node.upstream:
                        for upstream_node in basin_node.upstream:
                            if upstream_node not in basin_nodes:
                                basin_upstream.append(upstream_node)

                #print(" basin_upstream: ", basin_upstream)

                #print('\n')
                # here we do the numerical aggregation, actually changing the
                # internal values of the aggregating seg_node
                #summative_basin_area = self.net_upstream_area(current_node)
                new_pfaf_code = basin_pfaf_root

                current_node.aggregated_seg_ids.extend(aggregated_seg_ids)
                current_node.pfaf_code = new_pfaf_code
                current_node.basin_area += summative_basin_area
                if len(basin_upstream) > 0:
                    current_node.upstream = basin_upstream
                else:
                    current_node.upstream: List[SegNode] = list()
                aggregation_target_nodes.remove(current_node)

            # Here we need to update various properties of the flow tree
            # So that our changes are reflected in the net tree's attributes
            N = len(self.seg_id_values)
            self.adj_mat = self.reconstruct_adj_mat(node=self.outlet,adj_mat = np.zeros(shape=(N,N), dtype = int))
            self.network_graph = plotting.create_nxgraph(self.adj_mat)
            self.network_positions = plotting.organize_nxgraph(self.network_graph)

    def color_network_graph(self,measure,cmap):
        if type(measure) != type(None):
            return plotting.color_code_nxgraph(self.network_graph,measure,cmap)
        else:
            color_bar = None
            segs = np.arange(0,len(self.seg_id_values))
            color_vals = segs/len(segs)
            color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_vals, segs)}
            return color_dict, color_bar

    def size_network_graph(self,measure):
        segs = np.arange(0,len(self.seg_id_values))
        size_vals = segs/len(segs)
        size_dict = {f'{seg}': 200*size_vals(i) for i, seg in zip(size_vals,segs)}
        return size_dict

    def draw_network(self,label_map=[], color_measure=None, cmap = mpl.cm.get_cmap('hsv'),
                     node_size = 200, font_size = 8, font_weight = 'bold',
                     node_shape = 's', linewidths = 2, font_color = 'w', node_color = None,
                     with_labels=False,with_cbar=False,with_background=True,cbar_labelsize=10,
                     edge_color='k', alpha=1, cbar_title = '', cbar_label_pad=40):
        """
            draw_network
                plots the network through networkx
        """

        if type(color_measure) != type(None):
            if type(color_measure) != pd.Series:
                raise Exception("Color_measure is not a pandas Series")
            elif color_measure.size != len(self.seg_id_values):
                raise Exception("Color_measure size does not match number of nodes, double check measure aggregation.")

        network_color_dict, network_color_cbar = self.color_network_graph(color_measure,cmap)
        
        # we need to make sure that if the nodes have been relabeled by a previous
        # draw_network call, that we then restore them to their original labels
        # for future relabling

        expected_labels = list()
        current_labels = list()
        match = True
        for expected_label, current_label in zip(network_color_dict.keys(),self.network_graph.nodes):
            expected_labels.append(int(expected_label))
            current_labels.append(current_label)
            if match and (int(expected_label) != current_label):
                match = False
        if not match:
            # if the nodes have been relabeled from what we expect them to be
            # then we will adjust relabel them to their original values
            # since we have already checked that they are the same length
            standard_label_map = np.arange(len(expected_labels))
            new_network_graph = nx.relabel_nodes(self.network_graph,
                                             dict(zip(self.network_graph.nodes(),standard_label_map)),copy=True)
            self.network_graph = new_network_graph
            self.network_positions = plotting.organize_nxgraph(self.network_graph)        

        
        
        # if we want to relabel the nodes in this function call,
        # then we will do so here
        if len(label_map) > 0:
            new_network_color_dict = dict()
            for key in network_color_dict.keys():
                new_network_color_dict[f"{label_map[int(key)]}"] = network_color_dict[key]
            new_network_graph = nx.relabel_nodes(self.network_graph,
                                                 dict(zip(self.network_graph.nodes(),label_map)),copy=True)
            network_color_dict = new_network_color_dict
            self.network_graph = new_network_graph
            self.network_positions = plotting.organize_nxgraph(self.network_graph)

        network_nodecolors = [network_color_dict[f'{node}'] for node in self.network_graph.nodes()]
        if node_color:
            network_nodecolors = node_color

        nx.draw_networkx(self.network_graph,self.network_positions,with_labels=with_labels,
                         node_size=node_size,font_size=font_size,font_weight=font_weight,node_shape=node_shape,
                         linewidths=linewidths,font_color=font_color,node_color=network_nodecolors,
                         edge_color=edge_color, alpha=alpha)
        if with_cbar:
            cbar = plt.colorbar(network_color_cbar)
            cbar.ax.tick_params(labelsize=cbar_labelsize)
            cbar.set_label(cbar_title, rotation=270, labelpad=cbar_label_pad)
        if not with_background:
            plt.axis('off')
            
    def draw_multi_measure(self, color_dict, label_map = [], 
                     node_size = 200, font_size = 8, font_weight = 'bold', node_shape = 's', 
                     linewidths = 2, font_color = 'w', alpha = 1.0,
                     with_labels=False, with_cbar=False, with_background=True):
        """
            draw_multi_measure
                plots several networkx plots of user specified transparency for a single
                SimpleRiverNetwork to compare mutliple measures at once.
            ----
            color_dict:
                a dictionary set up as {name: [pd.Series,cmap, alpha]}
        """
        # first we will double check that the alpha's do not sum to more than 1, for this
        # may muddle the data

        alpha_sum = 0
        for color_key in color_dict.keys():
            alpha_sum += color_dict[color_key][2]
        
        if alpha_sum > 1.0:
            raise Exception("alpha values sum to more than 1.0, this may cover up data please revise")
        else:
            for color_key in color_dict.keys():
                self.draw_network(label_map = label_map, color_measure=color_dict[color_key][0], cmap = color_dict[color_key][1], 
                                  node_size=node_size, font_size=font_size, font_weight=font_weight, node_shape=node_shape, 
                                  linewidths=linewidths, font_color=font_color, alpha=color_dict[color_key][2], with_labels=with_labels, 
                                  with_cbar=with_cbar, with_background=with_background)

    def aggregate_measure_sum(self, dataset_seg_ids: np.ndarray, variable: np.ndarray)-> pd.Series:
        """
        aggregate_measure_sum
            determines the sum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(np.sum(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def aggregate_measure_mean(self, dataset_seg_ids: np.ndarray, variable: np.ndarray)-> pd.Series:
        """
        aggregate_measure_mean
            determines the mean measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(np.mean(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def aggregate_measure_median(self, dataset_seg_ids: np.ndarray, variable: np.ndarray)-> pd.Series:
        """
        aggregate_measure_median
            determines the median measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(np.median(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def aggregate_measure_max(self, dataset_seg_ids: np.ndarray, variable: np.ndarray)-> pd.Series:
        """
        aggregate_measure_max
            determines the maximum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(np.max(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def aggregate_measure_min(self, dataset_seg_ids: np.ndarray, variable: np.ndarray)-> pd.Series:
        """
        aggregate_measure_min
            determines the minimum measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(np.min(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def aggregate_measure(self, dataset_seg_ids: np.ndarray, variable: np.ndarray, aggregation_function)-> pd.Series:
        """
        aggregate_measure
            aggregates the measure value for the given variable based on how
            SimpleRiverNetwork has been aggregated and provides a pd.Series to plot on
            the SimpleRiverNetwork
        ----
        dataset_seg_ids:
            a np.ndarray containing all the seg_id values according to the original
            topology. this should be in the same order seg order as variable
        variable:
            a np.ndarray containing all the variable values according to the original
            topology. this should be in the same order seg order as dataset_seg_ids
        aggregation_function:
            a function to be passed in on how the aggregated segs should have this variable
            combined, recommended as a numpy function like np.sum, np.median, np.mean ...
        return:
            a pd.Series formated as: (seg_id_values_index, aggregated measure)
        """
        # color_measure, (for plotting) is formmated as (seg_id_values_index, value)

        if len(dataset_seg_ids.shape) != 1:
            raise Exception("The dimension of `dataset_seg_ids` is not 1, aggregation may be inaccurate")
        if len(variable.shape) != 1:
            raise Exception("The dimension of `variable` is not 1, aggregation may be inaccurate")

        new_measure_data = list()

        for seg_id in self.seg_id_values:
            node = self.find_node(seg_id, self.outlet)
            seg_id_index = np.where(dataset_seg_ids == seg_id)[0]
            measure_var_seg = [variable[seg_id_index]]
            for aggregated_seg_id in node.aggregated_seg_ids:
                aggregated_seg_id_index = np.where(dataset_seg_ids == aggregated_seg_id)[0]
                measure_var_seg.append(variable[aggregated_seg_id_index])
            new_measure_data.append(aggregation_function(measure_var_seg))

        return pd.Series(data = new_measure_data, index = np.arange(0,len(self.seg_id_values)))

    def spawn_srn(self, spawn_outlet):
        """
        spawn_srn
            creates a new SimpleRiverNetwork from that
            given network and upstream of it
        ----
        spawn_outlet:
            the seg_id of an outlet that the new tree
            is to be spawned from
        return:
            a new SimpleRiverNetwork
        """
        self.clear_end_markers
        spawn_segIDs = list()
        for node in spawn_outlet:
            spawn_segIDs.append(node.seg_id)

        spawn_seg_indexes = list()
        for segID in spawn_segIDs:
            seg_index = np.where(self.seg_id_values == segID)[0][0]
            spawn_seg_indexes.append(seg_index)

        spawn_topo = self.topo.isel(seg=spawn_seg_indexes)
        return SimpleRiverNetwork(spawn_topo,"")
