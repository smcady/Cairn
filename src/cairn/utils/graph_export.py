"""Export the idea graph as an interactive HTML visualization using force-graph."""

from __future__ import annotations

import html
import json
from pathlib import Path

from cairn.models.graph_types import EdgeType, IdeaGraph, NodeStatus, NodeType

# Node color by type
NODE_COLORS: dict[NodeType, str] = {
    NodeType.PROPOSITION: "#4a90d9",
    NodeType.QUESTION: "#e8922d",
    NodeType.TENSION: "#d94a4a",
    NodeType.EVIDENCE: "#4db870",
    NodeType.OBJECTION: "#d96aa8",
    NodeType.SYNTHESIS: "#8b5ecf",
    NodeType.FRAME: "#3fc1c9",
    NodeType.TERRITORY: "#888888",
    NodeType.ABSTRACTION: "#2ea8a0",
}

# Edge color by type
EDGE_COLORS: dict[EdgeType, str] = {
    EdgeType.SUPPORTS: "#4db870",
    EdgeType.CONTRADICTS: "#d94a4a",
    EdgeType.QUESTIONS: "#e8922d",
    EdgeType.RELATES_TO: "#888888",
    EdgeType.REFRAMES: "#3fc1c9",
    EdgeType.SYNTHESIZES: "#8b5ecf",
    EdgeType.ABSTRACTS_FROM: "#2ea8a0",
    EdgeType.RESOLVES: "#4a90d9",
    EdgeType.BETWEEN: "#aaaaaa",
    EdgeType.ADJACENT_TO: "#aaaaaa",
}

# Edge types that get animated particles
PARTICLE_EDGE_TYPES = {
    EdgeType.SUPPORTS,
    EdgeType.CONTRADICTS,
    EdgeType.SYNTHESIZES,
}

DIMMED_STATUSES = {NodeStatus.PARKED, NodeStatus.SUPERSEDED, NodeStatus.RESOLVED}


def _serialize_graph(graph: IdeaGraph) -> dict:
    """Serialize the IdeaGraph into a JSON-compatible dict for force-graph."""
    nodes = []
    for node in graph.get_all_nodes():
        nodes.append({
            "id": node.id,
            "text": node.text,
            "type": node.type.value,
            "confidence": node.confidence,
            "status": node.status.value,
            "depth": node.depth_of_exploration,
            "color": NODE_COLORS.get(node.type, "#888888"),
            "dimmed": node.status in DIMMED_STATUSES,
        })

    links = []
    for u, v, _key, data in graph.graph.edges(keys=True, data=True):
        edge_type_str = data.get("type", "RELATES_TO")
        try:
            edge_type = EdgeType(edge_type_str)
        except ValueError:
            edge_type = EdgeType.RELATES_TO
        links.append({
            "source": u,
            "target": v,
            "edgeType": edge_type.value,
            "label": edge_type.value.replace("_", " ").lower(),
            "color": EDGE_COLORS.get(edge_type, "#888888"),
            "particles": edge_type in PARTICLE_EDGE_TYPES,
        })

    return {"nodes": nodes, "links": links}


# Node type display labels for the legend
_NODE_TYPE_LABELS = {
    NodeType.PROPOSITION: "Proposition",
    NodeType.QUESTION: "Question",
    NodeType.TENSION: "Tension",
    NodeType.EVIDENCE: "Evidence",
    NodeType.OBJECTION: "Objection",
    NodeType.SYNTHESIS: "Synthesis",
    NodeType.FRAME: "Frame",
    NodeType.TERRITORY: "Territory",
    NodeType.ABSTRACTION: "Abstraction",
}


def export_graph_html(graph: IdeaGraph, output_path: Path) -> None:
    """Export the idea graph as a standalone interactive HTML file.

    Uses force-graph (WebGL-accelerated Canvas) loaded from CDN to produce
    a self-contained HTML file with pan/zoom/drag that opens in any browser.
    """
    graph_data = _serialize_graph(graph)
    graph_json = json.dumps(graph_data)

    # Build legend items JSON
    legend_items = json.dumps([
        {"type": label, "color": NODE_COLORS[nt]}
        for nt, label in _NODE_TYPE_LABELS.items()
    ])

    html_content = _HTML_TEMPLATE.replace("__GRAPH_DATA__", graph_json)
    html_content = html_content.replace("__LEGEND_ITEMS__", legend_items)

    output_path.write_text(html_content, encoding="utf-8")


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Idea Graph</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f1a;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e0e0e0;
  }
  #graph { width: 100vw; height: 100vh; }

  /* Hover panel */
  #hover-panel {
    display: none;
    position: fixed;
    top: 16px;
    right: 16px;
    width: 320px;
    background: rgba(20, 20, 40, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(12px);
    z-index: 10;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }
  #hover-panel .node-type {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    opacity: 0.7;
  }
  #hover-panel .node-text {
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 14px;
    word-wrap: break-word;
  }
  #hover-panel .meta {
    font-size: 12px;
    opacity: 0.5;
    line-height: 1.8;
  }
  #hover-panel .meta span { opacity: 1; }

  /* Legend */
  #legend {
    position: fixed;
    bottom: 16px;
    left: 16px;
    background: rgba(20, 20, 40, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 14px 18px;
    backdrop-filter: blur(12px);
    z-index: 10;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }
  #legend h3 {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.5;
    margin-bottom: 10px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
    font-size: 12px;
    opacity: 0.8;
  }
  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
</style>
</head>
<body>
<div id="graph"></div>

<div id="hover-panel">
  <div class="node-type" id="panel-type"></div>
  <div class="node-text" id="panel-text"></div>
  <div class="meta" id="panel-meta"></div>
</div>

<div id="legend">
  <h3>Node Types</h3>
  <div id="legend-items"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/force-graph"></script>
<script>
(function() {
  const data = __GRAPH_DATA__;
  const legendItems = __LEGEND_ITEMS__;

  // Build legend
  const legendContainer = document.getElementById('legend-items');
  legendItems.forEach(function(item) {
    const div = document.createElement('div');
    div.className = 'legend-item';
    const dot = document.createElement('div');
    dot.className = 'legend-dot';
    dot.style.background = item.color;
    dot.style.boxShadow = '0 0 6px ' + item.color + '88';
    const label = document.createElement('span');
    label.textContent = item.type;
    div.appendChild(dot);
    div.appendChild(label);
    legendContainer.appendChild(div);
  });

  // Hover panel elements
  const hoverPanel = document.getElementById('hover-panel');
  const panelType = document.getElementById('panel-type');
  const panelText = document.getElementById('panel-text');
  const panelMeta = document.getElementById('panel-meta');

  let hoveredNode = null;
  let hoveredLink = null;

  const graph = new ForceGraph(document.getElementById('graph'))
    .backgroundColor('#0f0f1a')

    // Node rendering
    .nodeCanvasObject(function(node, ctx, globalScale) {
      const r = 4 + node.confidence * 8;
      const alpha = node.dimmed ? 0.3 : 1.0;
      const isHovered = hoveredNode && hoveredNode.id === node.id;

      ctx.save();
      ctx.globalAlpha = alpha;

      // Glow
      if (!node.dimmed) {
        ctx.shadowColor = node.color;
        ctx.shadowBlur = isHovered ? 20 : 10;
      }

      // Circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Border
      ctx.shadowBlur = 0;
      ctx.strokeStyle = isHovered ? '#ffffff' : 'rgba(255,255,255,0.2)';
      ctx.lineWidth = isHovered ? 1.5 / globalScale : 0.5 / globalScale;
      ctx.stroke();

      // Label (only at reasonable zoom)
      if (globalScale > 0.6) {
        var fontSize = Math.max(3, 11 / globalScale);
        ctx.font = fontSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.shadowBlur = 0;
        ctx.fillStyle = node.dimmed ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.85)';

        var label = node.text.length > 40 ? node.text.substring(0, 40) + '...' : node.text;
        ctx.fillText(label, node.x, node.y + r + 2 / globalScale);
      }

      ctx.restore();
    })
    .nodePointerAreaPaint(function(node, color, ctx) {
      var r = 4 + node.confidence * 8;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    })

    // Links
    .linkColor(function(link) { return link.color; })
    .linkWidth(1.2)
    .linkDirectionalArrowLength(6)
    .linkDirectionalArrowRelPos(0.85)
    .linkCurvature(0.2)
    .linkDirectionalParticles(function(link) { return link.particles ? 3 : 0; })
    .linkDirectionalParticleWidth(2)
    .linkDirectionalParticleSpeed(0.005)
    .linkCanvasObjectMode(function() { return 'after'; })
    .linkCanvasObject(function(link, ctx, globalScale) {
      var src = link.source;
      var tgt = link.target;
      if (typeof src !== 'object' || typeof tgt !== 'object') return;

      // Dashed overlay for weak relationship types
      if (link.edgeType === 'RELATES_TO' || link.edgeType === 'ADJACENT_TO') {
        ctx.save();
        ctx.strokeStyle = link.color;
        ctx.lineWidth = 1.2 / globalScale;
        ctx.setLineDash([4 / globalScale, 4 / globalScale]);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.stroke();
        ctx.restore();
      }

      // Edge label at midpoint (only when zoomed in enough)
      if (globalScale > 1.2) {
        var mx = (src.x + tgt.x) / 2;
        var my = (src.y + tgt.y) / 2;
        // Offset slightly for curvature
        var dx = tgt.x - src.x;
        var dy = tgt.y - src.y;
        mx += -dy * 0.1;
        my += dx * 0.1;

        var fontSize = Math.max(2.5, 8 / globalScale);
        ctx.save();
        ctx.font = fontSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = link.color;
        ctx.globalAlpha = 0.7;
        ctx.fillText(link.label, mx, my);
        ctx.restore();
      }
    })

    // Hover interactions
    .onNodeHover(function(node) {
      hoveredNode = node;
      if (!node && !hoveredLink) {
        hoverPanel.style.display = 'none';
        document.getElementById('graph').style.cursor = 'default';
        return;
      }
      if (node) {
        hoveredLink = null;
        document.getElementById('graph').style.cursor = 'pointer';
        panelType.textContent = node.type;
        panelType.style.color = node.color;
        panelText.textContent = node.text;
        panelMeta.innerHTML =
          'Confidence: <span>' + node.confidence.toFixed(2) + '</span><br>' +
          'Status: <span>' + node.status + '</span><br>' +
          'Depth: <span>' + node.depth + '</span>';
        hoverPanel.style.display = 'block';
      }
    })
    .onLinkHover(function(link) {
      hoveredLink = link;
      if (!link && !hoveredNode) {
        hoverPanel.style.display = 'none';
        document.getElementById('graph').style.cursor = 'default';
        return;
      }
      if (link) {
        hoveredNode = null;
        document.getElementById('graph').style.cursor = 'pointer';
        var srcNode = link.source;
        var tgtNode = link.target;
        var srcLabel = (typeof srcNode === 'object') ? srcNode.text : srcNode;
        var tgtLabel = (typeof tgtNode === 'object') ? tgtNode.text : tgtNode;
        if (srcLabel && srcLabel.length > 50) srcLabel = srcLabel.substring(0, 50) + '...';
        if (tgtLabel && tgtLabel.length > 50) tgtLabel = tgtLabel.substring(0, 50) + '...';

        panelType.textContent = link.label;
        panelType.style.color = link.color;
        panelText.textContent = srcLabel + '  \u2192  ' + tgtLabel;
        panelMeta.innerHTML = 'Edge type: <span>' + link.edgeType + '</span>';
        hoverPanel.style.display = 'block';
      }
    });

  // Set data last so all config is applied before rendering
  graph.graphData(data);

  // d3-force tweaks (must access after graphData, since .strength() returns the force, not the graph)
  graph.d3Force('charge').strength(-200);
  graph.d3Force('link').distance(80);

  // Zoom to fit after layout settles
  setTimeout(function() { graph.zoomToFit(600, 60); }, 1500);
})();
</script>
</body>
</html>
"""
