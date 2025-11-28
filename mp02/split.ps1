param(
    [string]$Path,
    [int]$LineCount
)

$reader = New-Object System.IO.StreamReader($Path)
$writer1 = New-Object System.IO.StreamWriter("solution_part1.json")
$writer2 = New-Object System.IO.StreamWriter("solution_part2.json")

for ($i = 0; $i -lt $LineCount; $i++) {
    if ($reader.EndOfStream) { break }
    $writer1.WriteLine($reader.ReadLine())
}

while (-not $reader.EndOfStream) {
    $writer2.WriteLine($reader.ReadLine())
}

$reader.Close()
$writer1.Close()
$writer2.Close()
