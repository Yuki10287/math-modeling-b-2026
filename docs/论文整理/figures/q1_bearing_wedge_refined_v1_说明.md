# 问题一单次测向示意图美化

日期：2026-09-12。

图文件：[q1_bearing_wedge_refined_v1.png](q1_bearing_wedge_refined_v1.png)。由用户提供的原图通过内置图像生成工具编辑，作为论文插图候选；原图和论文正文未覆盖。

改动：移除绿色长引线，将 g 直接标在源点旁；统一蓝色误差边界、角度标注与数学字体；theta 改为正文中的 theta_i；调整线条层次与留白。

语义依据：[当前论文第2.1节](../问题一至三建模算法论文初稿.md)、[题面附录2](../../题目材料/B题题面_检索文本.md)。s_i 是测点，theta_i 是测得的示向度，g 为与观测相容的源位置，delta 是误差半宽。

建议图注：**单次测向的有界误差楔形示意图（误差角为便于辨识作放大展示，实际半宽为 δ=1°）。**

这是生成式位图示意，不用像素距离或曲线形状作几何计算；没有新增实验结果。

## 最终生成提示词

```text
Use case: scientific-educational, precise-object-edit.
Edit target: the attached bearing-angle geometry diagram. Produce one refined publication-quality diagram for a mathematical modeling paper. Preserve its mathematical meaning while making the composition harmonious and typography consistent. Plain pure white background, clean flat crisp vector-like line drawing, approximately 4:3 landscape, high resolution. No title, no legend, no decorative elements, no border, no watermark.

Required geometry:
- One small dark filled observation point at lower-left, labelled mathematical italic s with subscript i (s_i), label just below-left with comfortable padding.
- A horizontal baseline extending right from s_i in charcoal with a thinner stroke than the measured bearing.
- A straight charcoal measured-bearing ray with a small elegant filled arrowhead from s_i toward upper-right, about 40 degrees above horizontal.
- Exactly two muted blue dashed straight rays from the same s_i, symmetric around the central ray, roughly at 25 and 55 degrees above horizontal. Use consistent medium-long dashes and spacing. These represent angular error boundaries and must extend beyond the delta arcs. Do not draw any connecting outer curve, finite sector outline, shaded wedge, or filled region.
- A thin charcoal circular angle arc centered exactly on s_i from the horizontal baseline counterclockwise all the way to the central bearing ray, with a tiny arrow at the bearing end. Label theta with subscript i (θ_i) outside this arc, close to its low-angle portion and well below the lower blue ray to avoid crowding.
- Exactly two thin muted blue circular angle arcs, both precisely centered on s_i and on the same larger radius. One spans lower dashed ray to central ray, the other spans central ray to upper dashed ray. Each adjacent angular interval is labelled italic δ in blue, outside the arc at the respective interval midpoint. Balanced equal spacing, no overlaps. These arcs must meet their actual rays.
- A small muted sage-green filled point strictly inside the wedge, ABOVE the central bearing but BELOW the upper dashed boundary, and radially inside the blue angle arcs, visibly separated from all lines. This is the true source. Label italic g directly next to it, with a small gap, in matching sage green. REMOVE the original long green leader/arrow completely. Do not connect g to anything.

Typography: consistent classic serif math italic, LaTeX-like, well-formed subscripts and Greek letters. The ONLY visible labels are s_i, g, θ_i, δ, δ. All have coordinated size; subscripts smaller. Use charcoal #22272D for structural lines, restrained slate blue #5779A1 for error rays and delta marks, sage green #5C805F for g and point. Maintain adequate contrast for paper. Consistent disciplined stroke hierarchy: central bearing moderately strongest, baseline slightly lighter, blue dashed boundaries and angle arcs finer. Small restrained arrowheads.

Layout: diagram fills the page with generous balanced margins, approximately 10 percent margin around overall content. Vertex about 22 percent from left and 81 percent from top. Horizontal baseline endpoint near 87 percent from left; upper dashed tip near upper central-right. Compact balanced composition, no huge empty region. Professional mathematical journal figure; no 3D, texture, shadows, gradients, hand-drawn wobble, captions or additional text. The diagram intentionally exaggerates the error angles for readability; do not print any degree values.
```

