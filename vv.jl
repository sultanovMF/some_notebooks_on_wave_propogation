computing_area = ((i, j) for i in 1:5 for j in 1:5)


for (i, j) in computing_area
  println(i, ' ', j)
end
println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
for (i, j) in computing_area
  println(i, ' ', j)
end