#!/usr/bin/perl
package AI::ANN;
BEGIN {
  $AI::ANN::VERSION = '0.002';
}
use strict;
use warnings;
# ABSTRACT: an artificial neural network simulator

use AI::ANN::Neuron;
use Storable qw(dclone);


sub new {
	my $proto = shift;
	my $class = ref($proto) || $proto;
	my $self = {};
	my $data = shift;
	$self->{'inputcount'} = $data->{'inputs'};
	$self->{'outputneurons'} = [];
	$self->{'network'} = [];
	$self->{'inputs'} = [];
	$self->{'minvalue'} = $data->{'minvalue'} || 0;
	$self->{'maxvalue'} = $data->{'maxvalue'} || 1;
	$self->{'afunc'} = $data->{'afunc'} || sub { shift };
	my $neuronlist = $data->{'data'};
	for (my $i = 0; $i <= $#{$neuronlist} ; $i++) {
		push $self->{'outputneurons'}, $i # Requires Perl 5.14 !!!
			if $neuronlist->[$i]->{'iamanoutput'};
		$self->{'network'}->[$i]->{'object'} = 
			new AI::ANN::Neuron( 
				$i, 
				$neuronlist->[$i]->{'inputs'}, 
				$neuronlist->[$i]->{'neurons'} 
				);
	}
	bless($self, $class);
	return $self;
}


sub execute {
	my $self = shift;
	my $inputs = $self->{'inputs'} = shift;
	my $neurons = {};
	my $net = $self->{'network'}; # For less typing
	for (my $i = 0; $i <= $#{$net}; $i++) {
		delete $net->[$i]->{'done'};
		delete $net->[$i]->{'state'};
	}
	my $progress = 0;
	do {
		$progress = 0;
		for (my $i = 0; $i <= $#{$net}; $i++) {
			if ($net->[$i]->{'done'}) {next}
			if ($net->[$i]->{'object'}->ready($inputs, $neurons)) {
				my $potential = $net->[$i]->{'object'}->execute($inputs, $neurons);
				$potential = &{$self->{'afunc'}}($potential);
				$potential = $self->{'maxvalue'} if $potential > $self->{'maxvalue'};
				$potential = $self->{'minvalue'} if $potential < $self->{'minvalue'};
				$neurons->{$i} = $net->[$i]->{'state'} = $potential;
				$net->[$i]->{'done'} = 1;
				$progress++;
			}
		}
	} while ($progress); # If the network is feed-forward, we are now finished.
	
	my @notdone = grep {not (defined $net->[$_]->{'done'} &&
					$net->[$_]->{'done'} == 1)} 0..$#{$net};
	my %notdone; # Apparently Perl gets confused if I my the next line
	@notdone{@notdone} = undef; # %notdone is now a hash with a key for each 
								# 'notdone' neuron.
	if ($#notdone > 0) { #This is the part where we deal with loops and bad things
		my $maxerror = 0;
		my $loopcounter = 1;
		while (1) {
			foreach my $i (keys %notdone) { # Only bother iterating over the
											# ones we couldn't solve exactly
				# We don't care if it's ready now, we're just going to interate
				# until it stabilizes.
				if (not defined $neurons->{$i} && $i <= $#{$net}) {
					# Fixes warnings about uninitialized values, but we make 
					# sure $i is valid first.
					$neurons->{$i} = 0;
				}
				my $potential = $net->[$i]->{'object'}->execute($inputs, $neurons);
				$potential = &{$self->{'afunc'}}($potential);
				$potential = $self->{'maxvalue'} if $potential > $self->{'maxvalue'};
				$potential = $self->{'minvalue'} if $potential < $self->{'minvalue'};
				$notdone{$i} = $net->[$i]->{'state'} = $potential;
				# We want to know the absolute change
				if (abs($neurons->{$i}-$notdone{$i})>$maxerror) {
					$maxerror = abs($neurons->{$i}-$notdone{$i});
				}
			}
			foreach my $i (keys %notdone) { 
				# Update $neurons, since that is what gets passed to execute
				$neurons->{$i}=$notdone{$i};
			}
			if ($maxerror < 0.0001 && $loopcounter >= 5) {last}
			$loopcounter++;
			$maxerror=0;
		}
	}
	
	# Ok, hopefully all the neurons have happy values by now.
	# Get the output values for neurons corresponding to outputneurons
	my @output = map {$net->[$_]->{'state'}} @{$self->{'outputneurons'}};
	return \@output;
}


sub get_state {
	my $self = shift;
	my $net = $self->{'network'}; # For less typing
	my @neurons = map {$net->[$_]->{'state'}} 0..$#{$self->{'network'}};
	my @output = map {$net->[$_]->{'state'}} @{$self->{'outputneurons'}};
	
	return $self->{'inputs'}, \@neurons, \@output;
}


sub get_input_count {
	my $self = shift;
	return $self->{'inputcount'};
}


sub get_minvalue {
	my $self = shift;
	return $self->{'minvalue'};
}


sub get_maxvalue {
	my $self = shift;
	return $self->{'maxvalue'};
}


sub get_afunc {
	my $self = shift;
	return $self->{'afunc'};
}


sub get_internals {
	my $self = shift;
	my $retval = [];
	for (my $i = 0; $i <= $#{$self->{'network'}}; $i++) {
		$retval->[$i] = { iamanoutput => 0,
						  inputs => $self->{'network'}->[$i]->{'object'}->get_inputs(),
						  neurons => $self->{'network'}->[$i]->{'object'}->get_neurons()
						  };
	}
	foreach my $i (@{$self->{'outputneurons'}}) {
		$retval->[$i]->{'iamanoutput'} = 1;
	}
	return dclone($retval); # Dclone for safety.
}


sub readable {
	my $self = shift;
	my $retval = "This network has ". $self->{'inputcount'} ." inputs and ".
					scalar(@{$self->{'network'}}) ." neurons.\n";
	for (my $i = 0; $i <= $#{$self->{'network'}}; $i++) {
		$retval .= "Neuron $i\n";
		while (my ($k, $v) = each $self->{'network'}->[$i]->{'object'}->get_inputs()) {
			$retval .= "\tInput from input $k, weight is $v\n";
		}
		while (my ($k, $v) = each $self->{'network'}->[$i]->{'object'}->get_neurons()) {
			$retval .= "\tInput from neuron $k, weight is $v\n";
		}
		if (map {$_ == $i} $self->{'outputneurons'}) {
			$retval .= "\tThis neuron is a network output\n";
		}
	}
	return $retval;
}

1;

__END__
=pod

=head1 NAME

AI::ANN - an artificial neural network simulator

=head1 VERSION

version 0.002

=head1 SYNOPSIS

use AI::ANN;
my $network = new AI::ANN ( inputs => $inputcount, data => \@neuron_definition );
my $outputs = $network->execute( \@inputs );

=head1 METHODS

=head2 new

ANN::new({inputs => $inputcount, data => [{ iamanoutput => 0, inputs => {$inputid => $weight, ...}, neurons => {$neuronid => $weight}}, ...]})

inputs is number of inputs.
data is an arrayref of neuron definitions.
The first neuron with iamanoutput=1 is output 0. The second is output 1.
I hope you're seeing the pattern...
minvalue is the minimum value a neuron can pass. Default 0.
maxvalue is the maximum value a neuron can pass. Default 1.
afunc is a reference to the activation function. It should be simple and fast.
    The activation function is processed /before/ minvalue and maxvalue.

=head2 execute

$network->execute( [$input0, $input1, ...] )

Runs the network for as many iterations as necessary to achieve a stable
network, then returns the output. 
We store the current state of the network in two places - once in the object,
for persistence, and once in $neurons, for simplicity. This might be wrong, 
but I couldn't think of a better way.

=head2 get_state

$network->get_state()

Returns three arrayrefs, [$input0, ...], [$neuron0, ...], [$output0, ...], 
corresponding to the data from the last call to execute().
Intended primarily to assist with debugging.

=head2 get_input_count

$network->get_input_count()

Returns the number of inputs as a scalar.

=head2 get_minvalue

$network->get_minvalue()

Returns the minimum neuron value as a scalar.

=head2 get_maxvalue

$network->get_maxvalue()

Returns the maximum neuron value as a scalar.

=head2 get_afunc

$network->get_afunc()

Returns the activation function as a coderef.

=head2 get_internals

$network->get_internals()

Returns the weights in a not-human-consumable format.

=head2 readable

$network->readable()

Returns a human-friendly and diffable description of the network.

=head1 AUTHOR

Dan Collins <DCOLLINS@cpan.org>

=head1 COPYRIGHT AND LICENSE

This software is Copyright (c) 2011 by Dan Collins.

This is free software, licensed under:

  The GNU General Public License, Version 3, June 2007

=cut

