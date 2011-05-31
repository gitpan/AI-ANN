# change 'tests => 1' to 'tests => last_test_to_print';

use Test::More tests => 53;
BEGIN { use_ok('AI::ANN');
	use_ok('AI::ANN::Evolver'); };

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

# This test basically works by crossing a network with itself and making sure nothing changes

$network1=new AI::ANN ('inputs'=>1, 'data'=>[{ iamanoutput => 1, inputs => {0 => 1}, neurons => {}}]);
$network2=new AI::ANN ('inputs'=>1, 'data'=>[{ iamanoutput => 1, inputs => {0 => 1}, neurons => {}}]);

ok(defined $network1, "new() works");
ok($network1->isa("AI::ANN"), "Right class");
ok(defined $network2, "new() works");
ok($network2->isa("AI::ANN"), "Right class");

ok($out=$network1->execute([1]), "executed and still alive");

is($#{$out}, 0, "execute() output for a single neuron is the right length before crossover");
is($out->[0], 1, "execute() output for a single neuron has the correct value before crossover");

ok($out=$network2->execute([1]), "executed and still alive");

is($#{$out}, 0, "execute() output for a single neuron is the right length before crossover");
is($out->[0], 1, "execute() output for a single neuron has the correct value before crossover");

$evolver=new AI::ANN::Evolver ({});

ok(defined $evolver, "new() works");
ok($evolver->isa("AI::ANN::Evolver"), "Right class");

$network3=$evolver->crossover($network1, $network2);

ok(defined $network3, "crossover() works");
ok($network3->isa("AI::ANN"), "Right class");

ok($out=$network3->execute([1]), "executed and still alive");

is($#{$out}, 0, "execute() output for a single neuron is the right length after crossover");
is($out->[0], 1, "execute() output for a single neuron has the correct value after crossover");

($inputs, $neurons, $outputs) = $network3->get_state();

is($#{$inputs}, 0, "get_state inputs is correct length");
is($inputs->[0], 1, "get_state inputs returns correct element 0");

is($#{$neurons}, 0, "get_state neurons is correct length");
is($neurons->[0], 1, "get_state neurons returns correct element 0");

is($#{$outputs}, 0, "get_state outputs is correct length");
is($outputs->[0], 1, "get_state outputs returns correct element 0");

# Now we'll do it bigger!

$evolver=new AI::ANN::Evolver ({mutation_chance => 0.5, 
	mutation_amount => 0.2, add_link_chance => 0.2, 
	kill_link_chance => 0.2, sub_crossover_chance => 
	0.2, min_value => 0, max_value => 4});

$network1=new AI::ANN ('inputs'=>1,
		   'data'=>[{ iamanoutput => 0, inputs => {0 => 2}, neurons => {}},
                            { iamanoutput => 0, inputs => {0 => 1}, neurons => {0 => 1}},
                            { iamanoutput => 1, inputs => {}, neurons => {0 => 2, 1 => 1}}, 
                            { iamanoutput => 1, inputs => {}, neurons => {0 => 3}}]);
$network2=new AI::ANN ('inputs'=>1,
		   'data'=>[{ iamanoutput => 0, inputs => {0 => 1}, neurons => {}},
                            { iamanoutput => 0, inputs => {0 => 2}, neurons => {0 => 2}},
                            { iamanoutput => 1, inputs => {}, neurons => {0 => 3}}, 
                            { iamanoutput => 1, inputs => {}, neurons => {0 => 2, 1 => 1}}]);
ok(defined $network1, "new() works");
ok($network1->isa("AI::ANN"), "Right class");

ok($out=$network1->execute([1]), "executed and still alive");

is($#{$out}, 1, "execute() output is the right length");

($inputs, $neurons, $outputs) = $network1->get_state();

is($#{$inputs}, 0, "get_state inputs is correct length");
is($#{$neurons}, 3, "get_state neurons is correct length");
is($#{$outputs}, 1, "get_state outputs is correct length");

ok(defined $network2, "new() works");
ok($network2->isa("AI::ANN"), "Right class");

ok($out=$network2->execute([1]), "executed and still alive");

is($#{$out}, 1, "execute() output is the right length");

($inputs, $neurons, $outputs) = $network2->get_state();

is($#{$inputs}, 0, "get_state inputs is correct length");
is($#{$neurons}, 3, "get_state neurons is correct length");
is($#{$outputs}, 1, "get_state outputs is correct length");

$network3=$evolver->crossover($network1, $network2);

ok(defined $network3, "crossover() works");
ok($network3->isa("AI::ANN"), "Right class");

ok($out=$network3->execute([1]), "crossed over, executed and still alive");

is($#{$out}, 1, "execute() output after crossover is the right length");

($inputs, $neurons, $outputs) = $network3->get_state();

is($#{$inputs}, 0, "get_state inputs after crossover is correct length");
is($#{$neurons}, 3, "get_state neurons after crossover is correct length");
is($#{$outputs}, 1, "get_state outputs after crossover is correct length");

# Finally, we'll test mutate

$network4=$evolver->mutate($network1);

ok(defined $network4, "mutate() works");
ok($network4->isa("AI::ANN"), "Right class");

ok($out=$network4->execute([1]), "mutated, executed and still alive");

is($#{$out}, 1, "execute() output after mutate is the right length");

($inputs, $neurons, $outputs) = $network4->get_state();

is($#{$inputs}, 0, "get_state inputs after mutate is correct length");
is($#{$neurons}, 3, "get_state neurons after mutate is correct length");
is($#{$outputs}, 1, "get_state outputs after mutate is correct length");

