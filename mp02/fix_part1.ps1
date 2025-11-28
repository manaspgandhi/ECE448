param(
    [string]$Path
)

$tempFile = [System.IO.Path]::GetTempFileName()
$reader = New-Object System.IO.StreamReader($Path)
$writer = New-Object System.IO.StreamWriter($tempFile)

$lastLine = $null
$line = $null
while (-not $reader.EndOfStream) {
    if ($line -ne $null) {
        $writer.WriteLine($line)
    }
    $line = $reader.ReadLine()
}
$lastLine = $line

if ($lastLine -ne $null) {
    if ($lastLine.Trim().EndsWith(',')) {
        $lastLine = $lastLine.Substring(0, $lastLine.LastIndexOf(','))
    }
    $writer.WriteLine($lastLine)
}

$writer.WriteLine("        }")
$writer.WriteLine("    }")
$writer.WriteLine("}")

$reader.Close()
$writer.Close()

Move-Item -Path $tempFile -Destination $Path -Force
