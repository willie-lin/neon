Differential Layer
==================

File format
------------------

### Sections

A differential layer is a compact representation of a relational segment's WAL
 and page images.
The data is stored in several sections (not necessarily in this order):

1. The info page; which contains a version number and some information about
    the offsets to the next sections;
2. A new page versions map; which maps (local) page numbers to the location
    where their version data is stored;
3. An old page versions map; which contains a compressed map of (local) page 
    number to the LSN at which the page was last changed; but only if the
    page has no storage in the new page versions map, and has changed since
    the last full image layer;
4. A section that contains the actual changes to the pages, called the
    "lineages" section;

### Lineages

In the lineages section, the changes for each page are stored. The format is
optimized for replay efficiency: it is designed so that access to the latest
page versions is fast, and that any changes to the page are stored in a format
that is trivially replayable.

A Branch is a PageVersion as of some LSN, plus whatever WAL records that must
be applied on top of that PageVersion.

A Lineage is all Branches of a page in this layer; in order of decreasing LSN.
