#based on https://github.com/Glidias/mask2sam
#!/usr/bin/env python3

import math

# ported to python 3.x
# https://github.com/wsilva32/poly_decomp.py

def lineInt(l1, l2, precision=0):
    """Compute the intersection between two lines.

    Keyword arguments:
    l1 -- first line (list of two points: [[x1, y1], [x2, y2]])
    l2 -- second line
    precision -- precision to check if lines are parallel (default 0)

    Returns:
    The intersection point [x, y]
    """
    i = [0, 0]  # point
    a1 = l1[1][1] - l1[0][1]
    b1 = l1[0][0] - l1[1][0]
    c1 = a1 * l1[0][0] + b1 * l1[0][1]
    a2 = l2[1][1] - l2[0][1]
    b2 = l2[0][0] - l2[1][0]
    c2 = a2 * l2[0][0] + b2 * l2[0][1]
    det = a1 * b2 - a2 * b1
    if not scalar_eq(det, 0, precision):  # lines are not parallel
        i[0] = (b2 * c1 - b1 * c2) / det
        i[1] = (a1 * c2 - a2 * c1) / det
    return i


def lineSegmentsIntersect(p1, p2, q1, q2):
    """Checks if two line segments intersect.

    Keyword arguments:
    p1 -- The start vertex of the first line segment.
    p2 -- The end vertex of the first line segment.
    q1 -- The start vertex of the second line segment.
    q2 -- The end vertex of the second line segment.

    Returns:
    True if the two line segments intersect
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    da = q2[0] - q1[0]
    db = q2[1] - q1[1]

    # segments are parallel
    if (da * dy - db * dx) == 0:
        return False

    denom = da * dy - db * dx
    s = (dx * (q1[1] - p1[1]) + dy * (p1[0] - q1[0])) / denom
    t = (da * (p1[1] - q1[1]) + db * (q1[0] - p1[0])) / (-denom)

    return 0 <= s <= 1 and 0 <= t <= 1


def triangleArea(a, b, c):
    """Calculates the signed area of a triangle spanned by three points.
    Note that the area will be negative if the points are not given in counter-clockwise order.

    Keyword arguments:
    a -- First point [x, y]
    b -- Second point
    c -- Third point

    Returns:
    Signed area of triangle
    """
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((c[0] - a[0]) * (b[1] - a[1]))


def isLeft(a, b, c):
    return triangleArea(a, b, c) > 0


def isLeftOn(a, b, c):
    return triangleArea(a, b, c) >= 0


def isRight(a, b, c):
    return triangleArea(a, b, c) < 0


def isRightOn(a, b, c):
    return triangleArea(a, b, c) <= 0


def collinear(a, b, c, thresholdAngle=0):
    """Checks if three points are collinear.

    Keyword arguments:
    a -- First point
    b -- Second point
    c -- Third point
    thresholdAngle -- threshold to consider if points are collinear, in radians (default 0)

    Returns:
    True if points are collinear
    """
    if thresholdAngle == 0:
        return triangleArea(a, b, c) == 0
    else:
        ab = [b[0] - a[0], b[1] - a[1]]
        bc = [c[0] - b[0], c[1] - b[1]]

        dot = ab[0] * bc[0] + ab[1] * bc[1]
        magA = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1])
        magB = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1])
        if magA == 0 or magB == 0:
            return True
        cos_angle = dot / (magA * magB)
        # Clamp to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        return angle < thresholdAngle


def sqdist(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx * dx + dy * dy


def polygonAt(polygon, i):
    """Gets a vertex at position i on the polygon.
    It does not matter if i is out of bounds.

    Keyword arguments:
    polygon -- The polygon (list of [x, y] points)
    i -- Position desired on the polygon

    Returns:
    Vertex at position i
    """
    s = len(polygon)
    return polygon[i % s]


def polygonClear(polygon):
    """Clears the polygon data (in-place)."""
    polygon.clear()


def polygonAppend(polygon, poly, start, end):
    """Grabs points at indices `start` to `end` from `poly`
    and appends them to `polygon`.

    Keyword arguments:
    polygon -- The destination polygon (list)
    poly -- The source polygon (list)
    start -- Starting source index (inclusive)
    end -- Ending source index (exclusive)
    """
    for i in range(start, end):
        polygon.append(poly[i])


def polygonMakeCCW(polygon):
    """Makes sure that the polygon vertices are ordered counter-clockwise (in-place).

    Keyword arguments:
    polygon -- The polygon (list of [x, y])
    """
    br = 0
    v = polygon

    # find bottom right point
    for i in range(1, len(polygon)):
        if v[i][1] < v[br][1] or (v[i][1] == v[br][1] and v[i][0] > v[br][0]):
            br = i

    # reverse poly if clockwise
    if not isLeft(polygonAt(polygon, br - 1), polygonAt(polygon, br), polygonAt(polygon, br + 1)):
        polygonReverse(polygon)


def polygonReverse(polygon):
    """Reverses the vertices in the polygon (in-place)."""
    polygon.reverse()


def polygonIsReflex(polygon, i):
    """Checks if a point in the polygon is a reflex point.

    Keyword arguments:
    polygon -- The polygon
    i -- index of point to check

    Returns:
    True if point is a reflex point
    """
    return isRight(polygonAt(polygon, i - 1), polygonAt(polygon, i), polygonAt(polygon, i + 1))


def polygonCanSee(polygon, a, b):
    """Checks if two vertices in the polygon can see each other.

    Keyword arguments:
    polygon -- The polygon
    a -- Vertex index 1
    b -- Vertex index 2

    Returns:
    True if vertices can see each other
    """
    if isLeftOn(polygonAt(polygon, a + 1), polygonAt(polygon, a), polygonAt(polygon, b)) and \
       isRightOn(polygonAt(polygon, a - 1), polygonAt(polygon, a), polygonAt(polygon, b)):
        return False

    dist = sqdist(polygonAt(polygon, a), polygonAt(polygon, b))
    for i in range(len(polygon)):  # for each edge
        if (i + 1) % len(polygon) == a or i == a:  # ignore incident edges
            continue

        if isLeftOn(polygonAt(polygon, a), polygonAt(polygon, b), polygonAt(polygon, i + 1)) and \
           isRightOn(polygonAt(polygon, a), polygonAt(polygon, b), polygonAt(polygon, i)):
            # if diag intersects an edge
            l1 = [polygonAt(polygon, a), polygonAt(polygon, b)]
            l2 = [polygonAt(polygon, i), polygonAt(polygon, i + 1)]
            p = lineInt(l1, l2)
            if sqdist(polygonAt(polygon, a), p) < dist:  # if edge is blocking visibility to b
                return False

    return True


def polygonCopy(polygon, i, j, targetPoly=None):
    """Copies the polygon from vertex i to vertex j to targetPoly.

    Keyword arguments:
    polygon -- The source polygon
    i -- start vertex index
    j -- end vertex index (inclusive)
    targetPoly -- Optional target polygon (list)

    Returns:
    The resulting copy (list of points).
    """
    p = targetPoly if targetPoly is not None else []
    polygonClear(p)
    n = len(polygon)
    if i <= j:
        for k in range(i, j + 1):
            p.append(polygon[k])
    else:
        for k in range(i, n):
            p.append(polygon[k])
        for k in range(0, j + 1):
            p.append(polygon[k])
    return p


def polygonGetCutEdges(polygon):
    """Decomposes the polygon into convex pieces.
    Note that this algorithm has complexity O(N^4) and will be very slow for polygons with many vertices.

    Keyword arguments:
    polygon -- The polygon

    Returns:
    A list of edges [[p1,p2],[p2,p3],...] that cut the polygon.
    """
    mins = []
    nDiags = float('inf')
    tmpPoly = []

    for i in range(len(polygon)):
        if polygonIsReflex(polygon, i):
            for j in range(len(polygon)):
                if polygonCanSee(polygon, i, j):
                    tmp1 = polygonGetCutEdges(polygonCopy(polygon, i, j, tmpPoly))
                    tmp2 = polygonGetCutEdges(polygonCopy(polygon, j, i, tmpPoly))

                    combined = tmp1 + tmp2
                    if len(combined) < nDiags:
                        mins = combined
                        nDiags = len(combined)
                        mins.append([polygonAt(polygon, i), polygonAt(polygon, j)])

    return mins


def polygonDecomp(polygon):
    """Decomposes the polygon into one or more convex sub-polygons.

    Keyword arguments:
    polygon -- The polygon

    Returns:
    A list of convex polygons.
    """
    edges = polygonGetCutEdges(polygon)
    if edges:
        return polygonSlice(polygon, edges)
    else:
        return [polygon]


def polygonSlice(polygon, cutEdges):
    """Slices the polygon given one or more cut edges.

    Keyword arguments:
    polygon -- The polygon
    cutEdges -- A list of edges to cut on, as returned by getCutEdges()

    Returns:
    A list of polygon objects.
    """
    if not cutEdges:
        return [polygon]

    # Handle list of edges
    if isinstance(cutEdges, list) and cutEdges and isinstance(cutEdges[0], list):
        if len(cutEdges[0]) == 2 and isinstance(cutEdges[0][0], list):
            polys = [polygon[:]]  # make a copy

            for cutEdge in cutEdges:
                for j in range(len(polys)):
                    poly = polys[j]
                    result = polygonSlice(poly, cutEdge)
                    if result and len(result) == 2:
                        polys.pop(j)
                        polys.extend(result)
                        break
            return polys

    # Handle single edge
    cutEdge = cutEdges
    try:
        i = polygon.index(cutEdge[0])
        j = polygon.index(cutEdge[1])
    except ValueError:
        return False

    if i != -1 and j != -1:
        return [polygonCopy(polygon, i, j), polygonCopy(polygon, j, i)]
    else:
        return False


def polygonIsSimple(polygon):
    """Checks that the line segments of this polygon do not intersect each other.

    Keyword arguments:
    polygon -- The polygon

    Returns:
    True if polygon is simple (not self-intersecting)
    """
    n = len(polygon)
    if n < 3:
        return True

    # Check all pairs of non-adjacent edges
    for i in range(n):
        p1, p2 = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 2, n):
            # Skip adjacent edges (including closing edge)
            if (j == i) or ((j + 1) % n == i):
                continue
            q1, q2 = polygon[j], polygon[(j + 1) % n]
            if lineSegmentsIntersect(p1, p2, q1, q2):
                return False
    return True


def getIntersectionPoint(p1, p2, q1, q2, delta=0):
    """Gets the intersection point of two lines.

    Keyword arguments:
    p1 -- The start vertex of the first line segment.
    p2 -- The end vertex of the first line segment.
    q1 -- The start vertex of the second line segment.
    q2 -- The end vertex of the second line segment.
    delta -- Optional precision to check if lines are parallel (default 0)

    Returns:
    The intersection point [x, y].
    """
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = (a1 * p1[0]) + (b1 * p1[1])
    a2 = q2[1] - q1[1]
    b2 = q1[0] - q2[0]
    c2 = (a2 * q1[0]) + (b2 * q1[1])
    det = (a1 * b2) - (a2 * b1)

    if not scalar_eq(det, 0, delta):
        x = ((b2 * c1) - (b1 * c2)) / det
        y = ((a1 * c2) - (a2 * c1)) / det
        return [x, y]
    else:
        return [0, 0]


def polygonQuickDecomp(polygon, result=None, reflexVertices=None, steinerPoints=None, delta=25, maxlevel=100, level=0):
    """Quickly decompose the Polygon into convex sub-polygons.

    Keyword arguments:
    polygon -- The polygon to decompose
    result -- Stores result of decomposed polygon, passed recursively
    reflexVertices -- list to collect reflex vertices
    steinerPoints -- list to collect Steiner points
    delta -- Currently unused
    maxlevel -- The maximum allowed level of recursion
    level -- The current level of recursion

    Returns:
    List of decomposed convex polygons
    """
    if result is None:
        result = []
    if reflexVertices is None:
        reflexVertices = []
    if steinerPoints is None:
        steinerPoints = []

    poly = polygon
    n = len(poly)
    if n < 3:
        return result

    level += 1
    if level > maxlevel:
        print(f"quickDecomp: max level ({maxlevel}) reached.")
        return result

    for i in range(n):
        if polygonIsReflex(poly, i):
            reflexVertices.append(poly[i])
            upperDist = float('inf')
            lowerDist = float('inf')
            upperInt = [0, 0]
            lowerInt = [0, 0]
            upperIndex = -1
            lowerIndex = -1

            for j in range(n):
                # Lower chain
                if (isLeft(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j)) and
                    isRightOn(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j - 1))):
                    p = getIntersectionPoint(polygonAt(poly, i - 1), polygonAt(poly, i),
                                             polygonAt(poly, j), polygonAt(poly, j - 1))
                    if isRight(polygonAt(poly, i + 1), polygonAt(poly, i), p):
                        d = sqdist(poly[i], p)
                        if d < lowerDist:
                            lowerDist = d
                            lowerInt = p
                            lowerIndex = j

                # Upper chain
                if (isLeft(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j + 1)) and
                    isRightOn(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j))):
                    p = getIntersectionPoint(polygonAt(poly, i + 1), polygonAt(poly, i),
                                             polygonAt(poly, j), polygonAt(poly, j + 1))
                    if isLeft(polygonAt(poly, i - 1), polygonAt(poly, i), p):
                        d = sqdist(poly[i], p)
                        if d < upperDist:
                            upperDist = d
                            upperInt = p
                            upperIndex = j

            lowerPoly = []
            upperPoly = []

            if (lowerIndex + 1) % n == upperIndex:
                # Case 1: Insert Steiner point
                p_steiner = [(lowerInt[0] + upperInt[0]) / 2, (lowerInt[1] + upperInt[1]) / 2]
                steinerPoints.append(p_steiner)

                if i < upperIndex:
                    polygonAppend(lowerPoly, poly, i, upperIndex + 1)
                    lowerPoly.append(p_steiner)
                    upperPoly.append(p_steiner)
                    if lowerIndex != 0:
                        polygonAppend(upperPoly, poly, lowerIndex, n)
                    polygonAppend(upperPoly, poly, 0, i + 1)
                else:
                    if i != 0:
                        polygonAppend(lowerPoly, poly, i, n)
                    polygonAppend(lowerPoly, poly, 0, upperIndex + 1)
                    lowerPoly.append(p_steiner)
                    upperPoly.append(p_steiner)
                    polygonAppend(upperPoly, poly, lowerIndex, i + 1)
            else:
                # Case 2: Connect to closest visible vertex
                if lowerIndex > upperIndex:
                    upperIndex += n

                if upperIndex < lowerIndex:
                    result.append(polygon)
                    return result

                closestDist = float('inf')
                closestIndex = -1
                for j in range(lowerIndex, upperIndex + 1):
                    idx = j % n
                    if (isLeftOn(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, idx)) and
                        isRightOn(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, idx))):
                        d = sqdist(polygonAt(poly, i), polygonAt(poly, idx))
                        if d < closestDist:
                            closestDist = d
                            closestIndex = idx

                if closestIndex == -1:
                    result.append(polygon)
                    return result

                if i < closestIndex:
                    polygonAppend(lowerPoly, poly, i, closestIndex + 1)
                    if closestIndex != 0:
                        polygonAppend(upperPoly, poly, closestIndex, n)
                    polygonAppend(upperPoly, poly, 0, i + 1)
                else:
                    if i != 0:
                        polygonAppend(lowerPoly, poly, i, n)
                    polygonAppend(lowerPoly, poly, 0, closestIndex + 1)
                    polygonAppend(upperPoly, poly, closestIndex, i + 1)

            # Recurse on smaller polygon first
            if len(lowerPoly) < len(upperPoly):
                polygonQuickDecomp(lowerPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
                polygonQuickDecomp(upperPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
            else:
                polygonQuickDecomp(upperPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
                polygonQuickDecomp(lowerPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)

            return result

    result.append(polygon)
    return result


def polygonRemoveCollinearPoints(polygon, precision=0):
    """Remove collinear points in the polygon (in-place).

    Keyword arguments:
    polygon -- The polygon
    precision -- The threshold angle to use when determining whether two edges are collinear. (default is 0)

    Returns:
    The number of points removed
    """
    num = 0
    i = len(polygon) - 1
    while len(polygon) > 3 and i >= 0:
        if collinear(polygonAt(polygon, i - 1), polygonAt(polygon, i), polygonAt(polygon, i + 1), precision):
            del polygon[i % len(polygon)]
            num += 1
        i -= 1
    return num


def scalar_eq(a, b, precision=0):
    """Check if two scalars are equal within a given precision.

    Keyword arguments:
    a -- first scalar
    b -- second scalar
    precision -- precision to check equality

    Returns:
    True if scalars are equal within precision
    """
    return abs(a - b) <= precision
